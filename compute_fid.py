import os
import sys

import torch
from absl import flags
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import save_image

from diffusers.models import AutoencoderKL
from cleanfid import fid

from torchcfm.models.unet.unet import UNetModelWrapper
from scflow.data import ImageFolderWithPath
from scflow.inference import translate_tensor_ode


FLAGS = flags.FLAGS

# -------------------------
# Data / IO
# -------------------------
flags.DEFINE_string(
    "data_root",
    "/path/to/dataset",
    help="dataset root containing testA/testB",
)
flags.DEFINE_string(
    "direction",
    "AtoB",
    help="translation direction: 'AtoB' (testA->testB) or 'BtoA' (testB->testA)",
)
flags.DEFINE_string("ckpt_path", "", help="path to saved .pt checkpoint")
flags.DEFINE_string("out_dir", "./fid_runs/", help="where to write generated images")

# -------------------------
# Model type
# -------------------------
flags.DEFINE_string(
    "model",
    "otcfm",
    help="one of ['otcfm','icfm','fm','si','selfcfm']",
)

# -------------------------
# Mode: latent vs pixel
# -------------------------
flags.DEFINE_bool("latent", False, help="run in VAE latent space")
flags.DEFINE_bool("pixel", False, help="run directly in pixel space")

# -------------------------
# VAE (latent only)
# -------------------------
flags.DEFINE_string("vae_name", "stabilityai/sd-vae-ft-mse", help="diffusers VAE id")

# -------------------------
# UNet config
# -------------------------
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet (latent)")
flags.DEFINE_integer("num_res_blocks", 2, help="res blocks per level (latent)")
flags.DEFINE_string("attention_resolutions", "16", help="attention resolutions (latent)")

flags.DEFINE_integer("pixel_num_channel", 128, help="base channel of UNet (pixel)")
flags.DEFINE_integer("pixel_num_res_blocks", 2, help="res blocks per level (pixel)")
flags.DEFINE_string("pixel_attention_resolutions", "16", help="attention resolutions (pixel)")

# -------------------------
# Generation
# -------------------------
flags.DEFINE_integer("image_size", 256, help="input image size")
flags.DEFINE_integer("batch_size", 32, help="batch size for generation")
flags.DEFINE_integer("num_gen", 5000, help="how many images to generate/translate for FID")
flags.DEFINE_integer("integration_steps", 50, help="Euler steps if method=euler")
flags.DEFINE_string("integration_method", "dopri5", help="dopri5 or euler")
flags.DEFINE_float("tol", 1e-5, help="ODE solver tolerance")

# -------------------------
# Self-CFM / DINOv2 metadata
# -------------------------
flags.DEFINE_string(
    "backbone_name",
    "facebook/dinov2-base",
    help="for selfcfm: expected pairing backbone used during training (metadata/check only)",
)
flags.DEFINE_bool(
    "strict_cfg_check",
    False,
    help="if True, raise an error when ckpt cfg conflicts with command-line model/space/backbone",
)


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def parse_attention_resolutions(s):
    return s


def resolve_space():
    if FLAGS.pixel and FLAGS.latent:
        raise ValueError("Choose only one: --pixel or --latent")
    if FLAGS.pixel:
        return "pixel"
    return "latent"


def uses_conditioning():
    # In this repo:
    #   otcfm / icfm / si / selfcfm are conditioned translation models
    #   fm is unconditional generation
    return FLAGS.model in ["otcfm", "icfm", "si", "selfcfm"]


def load_state_dict_flexible(net, state_dict):
    try:
        net.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            new_sd[k.replace("module.", "")] = v
        net.load_state_dict(new_sd)


def maybe_warn_or_fail(msg):
    if FLAGS.strict_cfg_check:
        raise ValueError(msg)
    print(f"[Warning] {msg}")


def validate_ckpt_cfg(ckpt, requested_space):
    cfg = ckpt.get("cfg", None)
    if cfg is None:
        print("[Info] No cfg found in checkpoint; skipping config consistency checks.")
        return

    ckpt_model = cfg.get("model", None)
    ckpt_space = cfg.get("space", None)

    if ckpt_model is not None and ckpt_model != FLAGS.model:
        maybe_warn_or_fail(
            f"Checkpoint cfg model='{ckpt_model}' but command-line --model='{FLAGS.model}'."
        )

    if ckpt_space is not None and ckpt_space != requested_space:
        maybe_warn_or_fail(
            f"Checkpoint cfg space='{ckpt_space}' but requested space='{requested_space}'."
        )

    if FLAGS.model == "selfcfm":
        ckpt_backbone = cfg.get("backbone_name", None)
        if ckpt_backbone is not None and ckpt_backbone != FLAGS.backbone_name:
            maybe_warn_or_fail(
                f"Checkpoint cfg backbone_name='{ckpt_backbone}' but "
                f"command-line --backbone_name='{FLAGS.backbone_name}'. "
                f"This does not affect FID inference directly, but indicates you may be "
                f"evaluating a checkpoint from a different self-pairing setup."
            )

        if ckpt_space is not None and ckpt_space != "pixel":
            maybe_warn_or_fail(
                f"selfcfm checkpoint reports space='{ckpt_space}', but selfcfm FID inference expects pixel space."
            )


# -------------------------------------------------
# Main
# -------------------------------------------------
def main(argv):
    del argv

    if FLAGS.ckpt_path == "":
        raise ValueError("Please provide --ckpt_path")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    space = resolve_space()
    src, tgt = ("A", "B") if FLAGS.direction == "AtoB" else ("B", "A")

    if FLAGS.model == "selfcfm" and space != "pixel":
        raise ValueError("selfcfm is implemented only for pixel-space checkpoints in this version.")

    # -------------------------
    # Data
    # -------------------------
    tfm = transforms.Compose(
        [
            transforms.Resize(FLAGS.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(FLAGS.image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    testB_dir = os.path.join(FLAGS.data_root, f"test{tgt}")
    if not os.path.isdir(testB_dir):
        raise ValueError(f"Reference folder not found: {testB_dir}")

    loaderA_iter = None
    max_gen = FLAGS.num_gen

    if uses_conditioning():
        testA_dir = os.path.join(FLAGS.data_root, f"test{src}")
        if not os.path.isdir(testA_dir):
            raise ValueError(f"Conditioning folder not found: {testA_dir}")

        testA = ImageFolderWithPath(testA_dir, tfm)
        loaderA = torch.utils.data.DataLoader(
            testA,
            batch_size=FLAGS.batch_size,
            shuffle=False,
            num_workers=4,
            drop_last=False,
        )
        loaderA_iter = iter(loaderA)
        max_gen = min(FLAGS.num_gen, len(testA))

    # -------------------------
    # VAE
    # -------------------------
    vae = None
    if space == "latent":
        vae = AutoencoderKL.from_pretrained(FLAGS.vae_name).to(device)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

    # -------------------------
    # Model
    # -------------------------
    if space == "latent":
        net = UNetModelWrapper(
            dim=(4, 32, 32),
            num_res_blocks=FLAGS.num_res_blocks,
            num_channels=FLAGS.num_channel,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions=parse_attention_resolutions(FLAGS.attention_resolutions),
            dropout=0.1,
        ).to(device)
    else:
        net = UNetModelWrapper(
            dim=(3, FLAGS.image_size, FLAGS.image_size),
            num_res_blocks=FLAGS.pixel_num_res_blocks,
            num_channels=FLAGS.pixel_num_channel,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions=parse_attention_resolutions(FLAGS.pixel_attention_resolutions),
            dropout=0.1,
        ).to(device)

    ckpt = torch.load(FLAGS.ckpt_path, map_location=device)
    validate_ckpt_cfg(ckpt, space)

    state_dict = ckpt.get("ema_model", ckpt.get("net_model", ckpt))
    load_state_dict_flexible(net, state_dict)
    net.eval()

    # -------------------------
    # Output dir
    # -------------------------
    run_name = os.path.splitext(os.path.basename(FLAGS.ckpt_path))[0]
    gen_dir = os.path.join(FLAGS.out_dir, f"{FLAGS.model}_{space}_{run_name}")
    os.makedirs(gen_dir, exist_ok=True)

    print(f"[Info] Model       : {FLAGS.model}")
    print(f"[Info] Space       : {space}")
    print(f"[Info] Checkpoint  : {FLAGS.ckpt_path}")
    print(f"[Info] Output dir  : {gen_dir}")
    if FLAGS.model == "selfcfm":
        print(f"[Info] Pair backbone used in training (metadata/check): {FLAGS.backbone_name}")
        print("[Info] Note: DINOv2 pairing is a training-time module; FID generation uses only the learned flow UNet.")

    # -------------------------
    # Generation
    # -------------------------
    saved = 0
    pbar = tqdm(total=max_gen, desc=f"Generating ({FLAGS.model}, {space})")

    while saved < max_gen:
        if uses_conditioning():
            try:
                xA, paths = next(loaderA_iter)
            except StopIteration:
                break

            xA = xA.to(device)

            with torch.no_grad():
                if space == "latent":
                    zA = vae.encode(xA).latent_dist.mean
                    zB = translate_tensor_ode(
                        net,
                        zA,
                        steps=FLAGS.integration_steps,
                        method=FLAGS.integration_method,
                        tol=FLAGS.tol,
                    )
                    xB = vae.decode(zB).sample
                else:
                    xB = translate_tensor_ode(
                        net,
                        xA,
                        steps=FLAGS.integration_steps,
                        method=FLAGS.integration_method,
                        tol=FLAGS.tol,
                    )

        else:
            bsz = min(FLAGS.batch_size, max_gen - saved)
            with torch.no_grad():
                if space == "latent":
                    z0 = torch.randn(bsz, 4, 32, 32, device=device)
                    zB = translate_tensor_ode(
                        net,
                        z0,
                        steps=FLAGS.integration_steps,
                        method=FLAGS.integration_method,
                        tol=FLAGS.tol,
                    )
                    xB = vae.decode(zB).sample
                else:
                    x0 = torch.randn(bsz, 3, FLAGS.image_size, FLAGS.image_size, device=device)
                    xB = translate_tensor_ode(
                        net,
                        x0,
                        steps=FLAGS.integration_steps,
                        method=FLAGS.integration_method,
                        tol=FLAGS.tol,
                    )

        xB = (xB.clamp(-1, 1) + 1) / 2.0

        for i in range(xB.shape[0]):
            if saved >= max_gen:
                break

            if uses_conditioning():
                base = os.path.splitext(os.path.basename(paths[i]))[0]
                out_name = f"{base}_generated.png"
            else:
                out_name = f"{saved:06d}.png"

            save_image(xB[i], os.path.join(gen_dir, out_name))
            saved += 1
            pbar.update(1)

    pbar.close()

    # -------------------------
    # FID
    # -------------------------
    print("Computing FID:")
    print("  Generated:", gen_dir)
    print("  Reference:", testB_dir)

    score = fid.compute_fid(gen_dir, testB_dir, batch_size=FLAGS.batch_size)
    print("FID:", score)


if __name__ == "__main__":
    FLAGS(sys.argv)
    main(sys.argv)