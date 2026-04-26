import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers import AutoImageProcessor, AutoModel


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class PairingOutput:
    xB_perm: torch.Tensor
    pair_loss: torch.Tensor
    stats: Dict[str, torch.Tensor]
    assign_idx: torch.Tensor


class DINOSharedPairing(nn.Module):
    """
    Self-CFM pairing module:
      1. Frozen DINOv2 backbone -> patch tokens
      2. Trainable projector on patch tokens
      3. Global descriptors by mean-pooling projected patches
      4. Hard one-to-one assignment using Hungarian matching on global similarity
      5. Train projector using:
           - symmetric global InfoNCE on assigned pairs
           - patch alignment loss on assigned pairs
      6. Return reordered xB for downstream flow matching

    Input:
      xA, xB in [-1, 1], shape (B, 3, H, W)
    """

    def __init__(
        self,
        backbone_name: str = "facebook/dinov2-base",
        proj_dim: int = 256,
        proj_hidden_dim: int = 768,
        temperature: float = 0.07,
        conf_threshold: float = 0.20,
        lambda_global: float = 1.0,
        lambda_patch: float = 1.0,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.conf_threshold = conf_threshold
        self.lambda_global = lambda_global
        self.lambda_patch = lambda_patch

        self.processor = AutoImageProcessor.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name)

        hidden_dim = self.backbone.config.hidden_size
        self.patch_projector = MLP(hidden_dim, proj_hidden_dim, proj_dim)
        self.global_projector = MLP(hidden_dim, proj_hidden_dim, proj_dim)

        if freeze_backbone:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False

        mean = torch.tensor(self.processor.image_mean).view(1, 3, 1, 1)
        std = torch.tensor(self.processor.image_std).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean, persistent=False)
        self.register_buffer("img_std", std, persistent=False)

        crop_size = self.processor.crop_size
        if isinstance(crop_size, dict):
            self.backbone_size = int(crop_size["height"])
        else:
            self.backbone_size = int(crop_size)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # x in [-1,1] -> [0,1]
        x = (x + 1.0) / 2.0
        x = F.interpolate(
            x,
            size=(self.backbone_size, self.backbone_size),
            mode="bicubic",
            align_corners=False,
        )
        x = (x - self.img_mean) / self.img_std
        return x

    def _extract_tokens(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cls_token:   (B, D)
            patch_tokens:(B, N, D)
        """
        x = self._preprocess(x)

        if any(p.requires_grad for p in self.backbone.parameters()):
            out = self.backbone(pixel_values=x)
        else:
            with torch.no_grad():
                out = self.backbone(pixel_values=x)

        tokens = out.last_hidden_state  # (B, 1+N, D)
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        return cls_token, patch_tokens

    def _encode(self, x: torch.Tensor):
        cls_tok, patch_tok = self._extract_tokens(x)

        patch_z = self.patch_projector(patch_tok)          # (B, N, P)
        patch_z = F.normalize(patch_z, dim=-1)

        pooled = patch_tok.mean(dim=1) + cls_tok          # stronger global descriptor
        global_z = self.global_projector(pooled)          # (B, P)
        global_z = F.normalize(global_z, dim=-1)

        return global_z, patch_z

    @torch.no_grad()
    def _hungarian_assign(self, sim: torch.Tensor) -> torch.Tensor:
        """
        sim: (B, B), larger is better
        Returns:
            assign_idx: shape (B,), where xA[i] is paired with xB[assign_idx[i]]
        """
        row_ind, col_ind = linear_sum_assignment((-sim).detach().cpu().numpy())
        assign_idx = torch.full(
            (sim.shape[0],),
            fill_value=-1,
            device=sim.device,
            dtype=torch.long,
        )
        assign_idx[torch.as_tensor(row_ind, device=sim.device)] = torch.as_tensor(
            col_ind, device=sim.device, dtype=torch.long
        )
        return assign_idx

    def _global_contrastive_loss(
        self,
        zA: torch.Tensor,
        zB: torch.Tensor,
        assign_idx: torch.Tensor,
        sim: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Symmetric InfoNCE over assigned pairs.
        """
        B = zA.shape[0]
        row = torch.arange(B, device=zA.device)
        pos_sim = sim[row, assign_idx]

        valid = pos_sim >= self.conf_threshold
        num_valid = valid.sum()

        if num_valid.item() == 0:
            zero = torch.zeros((), device=zA.device, dtype=zA.dtype)
            stats = {
                "num_valid_pairs": torch.tensor(0, device=zA.device),
                "pair_conf_mean": pos_sim.mean().detach(),
            }
            return zero, stats

        logits_ab = torch.matmul(zA, zB.t()) / self.temperature
        logits_ba = logits_ab.t()

        loss_ab = F.cross_entropy(logits_ab[valid], assign_idx[valid])
        valid_cols = assign_idx[valid]
        target_rows = row[valid]
        loss_ba = F.cross_entropy(logits_ba[valid_cols], target_rows)

        loss = 0.5 * (loss_ab + loss_ba)

        stats = {
            "num_valid_pairs": num_valid.detach(),
            "pair_conf_mean": pos_sim.mean().detach(),
            "pair_conf_valid_mean": pos_sim[valid].mean().detach(),
        }
        return loss, stats

    def _patch_alignment_loss(
        self,
        patchA: torch.Tensor,
        patchB: torch.Tensor,
        assign_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        For each assigned pair, align local structure via patch-to-patch cosine similarity.
        Encourages each patch in A to find at least one good local counterpart in paired B,
        and vice versa.
        """
        patchB_perm = patchB[assign_idx]  # (B, N, P)

        # sim_patch: (B, N, N)
        sim_patch = torch.einsum("bnd,bmd->bnm", patchA, patchB_perm)

        best_a_to_b = sim_patch.max(dim=2).values.mean(dim=1)  # (B,)
        best_b_to_a = sim_patch.max(dim=1).values.mean(dim=1)  # (B,)

        patch_score = 0.5 * (best_a_to_b + best_b_to_a)
        patch_loss = 1.0 - patch_score.mean()
        return patch_loss

    def forward(self, xA: torch.Tensor, xB: torch.Tensor) -> PairingOutput:
        zA, patchA = self._encode(xA)
        zB, patchB = self._encode(xB)

        sim = torch.matmul(zA, zB.t())  # cosine similarity because normalized
        assign_idx = self._hungarian_assign(sim)

        xB_perm = xB[assign_idx]

        global_loss, global_stats = self._global_contrastive_loss(zA, zB, assign_idx, sim)
        patch_loss = self._patch_alignment_loss(patchA, patchB, assign_idx)

        pair_loss = self.lambda_global * global_loss + self.lambda_patch * patch_loss

        with torch.no_grad():
            B = xA.shape[0]
            row = torch.arange(B, device=xA.device)
            assigned_sim = sim[row, assign_idx]
            stats = {
                "pair_loss_global": global_loss.detach(),
                "pair_loss_patch": patch_loss.detach(),
                "pair_sim_mean": assigned_sim.mean().detach(),
                "pair_sim_min": assigned_sim.min().detach(),
                "pair_sim_max": assigned_sim.max().detach(),
                **global_stats,
            }

        return PairingOutput(
            xB_perm=xB_perm,
            pair_loss=pair_loss,
            stats=stats,
            assign_idx=assign_idx,
        )
