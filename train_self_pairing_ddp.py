# train_self_pairing_ddp.py
import copy
import math
import os

import torch
from absl import app, flags
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torchvision import transforms
from tqdm import trange

from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.unet.unet import UNetModelWrapper

from scflow.data import ImageFolderNoLabel
from scflow.inference import translate_and_save_grid
from scflow.pairing import DINOSharedPairing
from scflow.training import ema, infiniteloop, setup, unwrap_model


FLAGS = flags.FLAGS

# -------------------------
# Data
# -------------------------
flags.DEFINE_string(
    "data_root",
    "/path/to/dataset",
    help="dataset root containing trainA/trainB/testA/testB",
)
flags.DEFINE_string(
    "direction",
    "AtoB",
    help="translation direction: 'AtoB' (trainA->trainB) or 'BtoA' (trainB->trainA)",
)
flags.DEFINE_integer("image_size", 256, help="input image size")
flags.DEFINE_integer("num_workers", 4, help="workers of DataLoader")
flags.DEFINE_integer("batch_size", 64, help="global batch size across GPUs")

# -------------------------
# Method
# -------------------------
flags.DEFINE_string(
    "model",
    "selfcfm",
    help="for this file use 'selfcfm'",
)

# -------------------------
# Pixel only
# -------------------------
flags.DEFINE_bool("pixel", True, help="selfcfm is pixel-space only in this trainer")

# -------------------------
# Flow model
# -------------------------
flags.DEFINE_float("lr", 2e-4, help="flow model learning rate")
flags.DEFINE_float("pair_lr", 1e-4, help="pairing module learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer("total_steps", 200001, help="total training steps")
flags.DEFINE_integer("warmup", 5000, help="lr warmup steps")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay")
flags.DEFINE_float("sigma", 0.0, help="CFM sigma")
flags.DEFINE_float("pair_loss_weight", 0.10, help="weight for pairing loss")
flags.DEFINE_integer(
    "pair_warmup_steps",
    5000,
    help="before this step, FM still uses original xB; pairing module still trains",
)

# -------------------------
# UNet
# -------------------------
flags.DEFINE_integer("pixel_num_channel", 128, help="base channel of UNet")
flags.DEFINE_integer("pixel_num_res_blocks", 2, help="res blocks per level")
flags.DEFINE_string("pixel_attention_resolutions", "16", help="attention resolutions")

# -------------------------
# Pairing module
# -------------------------
flags.DEFINE_string("backbone_name", "facebook/dinov2-base", help="HF DINOv2 backbone")
flags.DEFINE_integer("pair_proj_dim", 256, help="projection dim")
flags.DEFINE_integer("pair_proj_hidden_dim", 768, help="projection hidden dim")
flags.DEFINE_float("pair_temperature", 0.07, help="contrastive temperature")
flags.DEFINE_float("pair_conf_threshold", 0.20, help="min cosine sim for valid positive")
flags.DEFINE_float("pair_lambda_global", 1.0, help="global contrastive weight")
flags.DEFINE_float("pair_lambda_patch", 1.0, help="patch alignment weight")
flags.DEFINE_bool("freeze_backbone", True, help="freeze DINOv2 backbone")

# -------------------------
# DDP
# -------------------------
flags.DEFINE_bool("parallel", False, help="multi-gpu training")
flags.DEFINE_string("master_addr", "localhost", help="master address for DDP")
flags.DEFINE_string("master_port", "12355", help="master port for DDP")

# -------------------------
# Saving / Eval
# -------------------------
flags.DEFINE_string("output_dir", "./results_selfcfm/", help="output directory")
flags.DEFINE_integer("save_step", 10000, help="checkpoint frequency")
flags.DEFINE_integer("sample_n", 8, help="preview count")


def warmup_scale(step: int):
    return min(step, FLAGS.warmup) / float(max(1, FLAGS.warmup))


def parse_attention_resolutions(s):
    return s


def train(rank, total_num_gpus, argv):
    del argv

    if FLAGS.model != "selfcfm":
        raise ValueError("This trainer is only for --model selfcfm")
    if not FLAGS.pixel:
        raise ValueError("selfcfm trainer is pixel-space only. Use --pixel True")

    is_main = (rank == 0) if FLAGS.parallel else True

    if FLAGS.parallel and total_num_gpus > 1:
        if FLAGS.batch_size % total_num_gpus != 0:
            raise ValueError("batch_size must be divisible by world size")
        batch_size_per_gpu = FLAGS.batch_size // total_num_gpus
        setup()
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        batch_size_per_gpu = FLAGS.batch_size
        device = rank if isinstance(rank, torch.device) else torch.device(f"cuda:{rank}")

    if is_main:
        print("Training SELF-CFM in pixel space")
        print("data_root:", FLAGS.data_root)
        print("lr:", FLAGS.lr, "pair_lr:", FLAGS.pair_lr)
        print("total_steps:", FLAGS.total_steps, "save_step:", FLAGS.save_step)
        print("sigma:", FLAGS.sigma)
        print("pair_warmup_steps:", FLAGS.pair_warmup_steps)
        print("backbone_name:", FLAGS.backbone_name)

    tfm = transforms.Compose([
        transforms.Resize(FLAGS.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(FLAGS.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    src, tgt = ("A", "B") if FLAGS.direction == "AtoB" else ("B", "A")
    trainA = ImageFolderNoLabel(os.path.join(FLAGS.data_root, f"train{src}"), tfm)
    trainB = ImageFolderNoLabel(os.path.join(FLAGS.data_root, f"train{tgt}"), tfm)

    samplerA = DistributedSampler(trainA) if FLAGS.parallel else None
    samplerB = DistributedSampler(trainB) if FLAGS.parallel else None

    loaderA = torch.utils.data.DataLoader(
        trainA,
        batch_size=batch_size_per_gpu,
        sampler=samplerA,
        shuffle=False if FLAGS.parallel else True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    loaderB = torch.utils.data.DataLoader(
        trainB,
        batch_size=batch_size_per_gpu,
        sampler=samplerB,
        shuffle=False if FLAGS.parallel else True,
        num_workers=FLAGS.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    loopA = infiniteloop(loaderA)
    loopB = infiniteloop(loaderB)

    if is_main:
        testA = ImageFolderNoLabel(os.path.join(FLAGS.data_root, f"test{src}"), tfm)
        testA_loader = torch.utils.data.DataLoader(
            testA,
            batch_size=FLAGS.sample_n,
            shuffle=False,
            num_workers=FLAGS.num_workers,
            drop_last=False,
            pin_memory=True,
        )
        fixed_testA = next(iter(testA_loader)).to(device)

    # -------------------------
    # Flow model
    # -------------------------
    net_model = UNetModelWrapper(
        dim=(3, FLAGS.image_size, FLAGS.image_size),
        num_res_blocks=FLAGS.pixel_num_res_blocks,
        num_channels=FLAGS.pixel_num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions=parse_attention_resolutions(FLAGS.pixel_attention_resolutions),
        dropout=0.1,
    ).to(device)

    ema_model = copy.deepcopy(net_model)

    # -------------------------
    # Pairing module
    # -------------------------
    pair_model = DINOSharedPairing(
        backbone_name=FLAGS.backbone_name,
        proj_dim=FLAGS.pair_proj_dim,
        proj_hidden_dim=FLAGS.pair_proj_hidden_dim,
        temperature=FLAGS.pair_temperature,
        conf_threshold=FLAGS.pair_conf_threshold,
        lambda_global=FLAGS.pair_lambda_global,
        lambda_patch=FLAGS.pair_lambda_patch,
        freeze_backbone=FLAGS.freeze_backbone,
    ).to(device)

    # -------------------------
    # Optimizer / schedulers
    # -------------------------
    flow_params = list(net_model.parameters())
    pair_params = [p for p in pair_model.parameters() if p.requires_grad]

    optim = torch.optim.Adam(
        [
            {"params": flow_params, "lr": FLAGS.lr},
            {"params": pair_params, "lr": FLAGS.pair_lr},
        ]
    )
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_scale)

    if FLAGS.parallel:
        net_model = DistributedDataParallel(net_model, device_ids=[device.index])
        ema_model = DistributedDataParallel(ema_model, device_ids=[device.index])
        pair_model = DistributedDataParallel(pair_model, device_ids=[device.index])

    if is_main:
        flow_count = sum(p.numel() for p in unwrap_model(net_model).parameters())
        pair_count = sum(p.numel() for p in unwrap_model(pair_model).parameters() if p.requires_grad)
        print(f"Flow params: {flow_count/1e6:.2f} M")
        print(f"Pairing trainable params: {pair_count/1e6:.2f} M")

    FM = ConditionalFlowMatcher(sigma=FLAGS.sigma)

    savedir = os.path.join(
        FLAGS.output_dir,
        f"selfcfm_pixel_singlegpu" if not FLAGS.parallel else f"selfcfm_pixel_ddp{total_num_gpus}gpu",
    )
    if is_main:
        os.makedirs(savedir, exist_ok=True)

    min_dataset_len = min(len(trainA), len(trainB))
    steps_per_epoch = math.ceil(min_dataset_len / batch_size_per_gpu)

    global_step = 0

    with trange(FLAGS.total_steps, dynamic_ncols=True, disable=not is_main) as pbar:
        epoch = 0
        while global_step < FLAGS.total_steps:
            if samplerA is not None:
                samplerA.set_epoch(epoch)
            if samplerB is not None:
                samplerB.set_epoch(epoch)

            for _ in range(steps_per_epoch):
                if global_step >= FLAGS.total_steps:
                    break

                net_model.train()
                pair_model.train()
                optim.zero_grad(set_to_none=True)

                xA = next(loopA).to(device, non_blocking=True)
                xB = next(loopB).to(device, non_blocking=True)

                pair_out = pair_model(xA, xB)
                xB_pair = pair_out.xB_perm
                pair_loss = pair_out.pair_loss

                # Warmup: let pairing module learn first, but avoid using early noisy pairs for FM
                if global_step < FLAGS.pair_warmup_steps:
                    xB_for_fm = xB
                else:
                    xB_for_fm = xB_pair

                t, xt, ut = FM.sample_location_and_conditional_flow(xA, xB_for_fm)
                vt = net_model(t, xt)
                fm_loss = torch.mean((vt - ut) ** 2)

                loss = fm_loss + FLAGS.pair_loss_weight * pair_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(unwrap_model(net_model).parameters(), FLAGS.grad_clip)
                torch.nn.utils.clip_grad_norm_(unwrap_model(pair_model).parameters(), FLAGS.grad_clip)

                optim.step()
                sched.step()
                ema(net_model, ema_model, FLAGS.ema_decay)

                if is_main:
                    stats = pair_out.stats
                    pbar.set_postfix({
                        "loss": float(loss.item()),
                        "fm": float(fm_loss.item()),
                        "pair": float(pair_loss.item()),
                        "sim": float(stats["pair_sim_mean"].item()),
                        "valid": int(stats["num_valid_pairs"].item()),
                    })
                    pbar.update(1)

                    if FLAGS.save_step > 0 and global_step % FLAGS.save_step == 0:
                        preview_net = unwrap_model(net_model)
                        preview_ema = unwrap_model(ema_model)

                        translate_and_save_grid(
                            model=preview_net,
                            vae=None,
                            xA=fixed_testA,
                            out_path=os.path.join(savedir, f"normal_translate_step_{global_step}.png"),
                        )
                        translate_and_save_grid(
                            model=preview_ema,
                            vae=None,
                            xA=fixed_testA,
                            out_path=os.path.join(savedir, f"ema_translate_step_{global_step}.png"),
                        )

                        ckpt = {
                            "net_model": net_model.state_dict(),
                            "ema_model": ema_model.state_dict(),
                            "pair_model": pair_model.state_dict(),
                            "sched": sched.state_dict(),
                            "optim": optim.state_dict(),
                            "step": global_step,
                            "cfg": {
                                "model": "selfcfm",
                                "space": "pixel",
                                "image_size": FLAGS.image_size,
                                "sigma": FLAGS.sigma,
                                "backbone_name": FLAGS.backbone_name,
                                "pair_proj_dim": FLAGS.pair_proj_dim,
                                "pair_proj_hidden_dim": FLAGS.pair_proj_hidden_dim,
                                "pair_temperature": FLAGS.pair_temperature,
                                "pair_conf_threshold": FLAGS.pair_conf_threshold,
                                "pair_lambda_global": FLAGS.pair_lambda_global,
                                "pair_lambda_patch": FLAGS.pair_lambda_patch,
                                "pair_loss_weight": FLAGS.pair_loss_weight,
                                "pair_warmup_steps": FLAGS.pair_warmup_steps,
                            },
                        }
                        torch.save(
                            ckpt,
                            os.path.join(savedir, f"selfcfm_pixel_step_{global_step}.pt"),
                        )

                global_step += 1

            epoch += 1

    if FLAGS.parallel:
        torch.distributed.destroy_process_group()


def main(argv):
    total_num_gpus = int(os.getenv("WORLD_SIZE", 1))

    if FLAGS.parallel and total_num_gpus > 1:
        rank = int(os.getenv("LOCAL_RANK", 0))
        train(rank=rank, total_num_gpus=total_num_gpus, argv=argv)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train(rank=device, total_num_gpus=total_num_gpus, argv=argv)


if __name__ == "__main__":
    app.run(main)