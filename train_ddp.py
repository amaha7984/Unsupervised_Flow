#train_ddp.py
import copy
import math
import os
import glob

import torch
from absl import app, flags
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler
from torchvision import transforms
from PIL import Image
from tqdm import trange

from diffusers.models import AutoencoderKL

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
)
from torchcfm.models.unet.unet import UNetModelWrapper

from utils import ema, infiniteloop, translate_and_save_grid


FLAGS = flags.FLAGS

# -------------------------
# Data
# -------------------------
flags.DEFINE_string(
    "data_root",
    "/path/to/dataset",
    help="dataset root containing trainA/trainB/testA/testB",
)
flags.DEFINE_integer("image_size", 256, help="input image size")
flags.DEFINE_integer("num_workers", 4, help="workers of Dataloader")
flags.DEFINE_integer("batch_size", 64, help="batch size (total across GPUs)")

# -------------------------
# Method
# -------------------------
flags.DEFINE_string("model", "otcfm", help="flow matching model type: one of ['otcfm','icfm','fm','si']")

# -------------------------
# Mode: latent vs pixel
# -------------------------
flags.DEFINE_bool("latent", False, help="run in VAE latent space (default when neither flag set)")
flags.DEFINE_bool("pixel", False, help="run directly in pixel space")

# -------------------------
# Model/Training
# -------------------------
flags.DEFINE_float("lr", 2e-4, help="target learning rate")
flags.DEFINE_float("grad_clip", 1.0, help="gradient norm clipping")
flags.DEFINE_integer("total_steps", 200001, help="total training steps")
flags.DEFINE_integer("warmup", 5000, help="learning rate warmup")
flags.DEFINE_float("ema_decay", 0.9999, help="ema decay rate")

# -------------------------
# OT-CFM specifics
# -------------------------
flags.DEFINE_float("sigma", 0.0, help="sigma for probability path (0.0 = deterministic path)")

# -------------------------
# VAE (only used for latent)
# -------------------------
flags.DEFINE_string("vae_name", "stabilityai/sd-vae-ft-mse", help="diffusers VAE id")

# -------------------------
# UNet (latent defaults)
# -------------------------
flags.DEFINE_integer("num_channel", 128, help="base channel of UNet")
flags.DEFINE_integer("num_res_blocks", 2, help="res blocks per level")
flags.DEFINE_string("attention_resolutions", "16", help="attention resolutions")

# -------------------------
# UNet (pixel overrides)
# -------------------------
flags.DEFINE_integer("pixel_num_channel", 128, help="base channel of UNet for pixel space")
flags.DEFINE_integer("pixel_num_res_blocks", 2, help="res blocks per level for pixel space")
flags.DEFINE_string("pixel_attention_resolutions", "16", help="attention resolutions for pixel space")

# -------------------------
# DDP
# -------------------------
flags.DEFINE_bool("parallel", False, help="multi gpu training")
flags.DEFINE_string("master_addr", "localhost", help="master address for DDP")
flags.DEFINE_string("master_port", "12355", help="master port for DDP")

# -------------------------
# Saving/Eval
# -------------------------
flags.DEFINE_string("output_dir", "./results_horse2zebra_vae/", help="output_directory")
flags.DEFINE_integer("save_step", 10000, help="frequency of saving checkpoints")
flags.DEFINE_integer("sample_n", 8, help="how many testA images to translate for preview grids")


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def setup(backend="nccl"):
    torch.distributed.init_process_group(backend=backend, init_method="env://")


class ImageFolderNoLabel(torch.utils.data.Dataset):

    def __init__(self, folder, transform):
        self.folder = folder
        self.transform = transform
        exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(folder, e)))
        files = sorted(files)
        if len(files) == 0:
            raise RuntimeError(f"No images found in: {folder}")
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)


def warmup_lr(step):
    return min(step, FLAGS.warmup) / float(FLAGS.warmup)


def resolve_space():
    if FLAGS.pixel and FLAGS.latent:
        raise ValueError("Choose only one: --pixel OR --latent")
    if FLAGS.pixel:
        return "pixel"
    return "latent"


def parse_attention_resolutions(s):
    return s


def build_flow_matcher():
    sigma = FLAGS.sigma
    if FLAGS.model == "otcfm":
        return ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "icfm":
        return ConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "fm":
        return TargetConditionalFlowMatcher(sigma=sigma)
    elif FLAGS.model == "si":
        return VariancePreservingConditionalFlowMatcher(sigma=sigma)
    else:
        raise NotImplementedError(
            f"Unknown model {FLAGS.model}, must be one of ['otcfm','icfm','fm','si']"
        )


# -------------------------------------------------
# Training
# -------------------------------------------------

def train(rank, total_num_gpus, argv):
    space = resolve_space()
    is_main = (rank == 0) if FLAGS.parallel else True

    if is_main:
        if space == "latent":
            print(f"Training {FLAGS.model.upper()} (unpaired A->B) in SD-VAE LATENT space")
        else:
            print(f"Training {FLAGS.model.upper()} (unpaired A->B) in PIXEL space")
        print("data_root:", FLAGS.data_root)
        print("lr, total_steps, ema_decay, save_step:", FLAGS.lr, FLAGS.total_steps, FLAGS.ema_decay, FLAGS.save_step)
        print("sigma:", FLAGS.sigma)
        if space == "latent":
            print("vae_name:", FLAGS.vae_name)

    if FLAGS.parallel and total_num_gpus > 1:
        batch_size_per_gpu = FLAGS.batch_size // total_num_gpus
        setup()  # uses env:// — torchrun already set all env vars
        torch.cuda.set_device(rank)
    else:
        batch_size_per_gpu = FLAGS.batch_size

    # -------------------------
    # Data
    # -------------------------
    tfm = transforms.Compose([
        transforms.Resize(FLAGS.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(FLAGS.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainA = ImageFolderNoLabel(os.path.join(FLAGS.data_root, "trainA"), tfm)
    trainB = ImageFolderNoLabel(os.path.join(FLAGS.data_root, "trainB"), tfm)

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
        testA = ImageFolderNoLabel(os.path.join(FLAGS.data_root, "testA"), tfm)
        testA_loader = torch.utils.data.DataLoader(
            testA,
            batch_size=FLAGS.sample_n,
            shuffle=False,
            num_workers=FLAGS.num_workers,
            drop_last=False,
            pin_memory=True,
        )

    # -------------------------
    # VAE (frozen, on each rank)
    # -------------------------
    vae = None
    if space == "latent":
        vae = AutoencoderKL.from_pretrained(FLAGS.vae_name).to(rank)
        vae.eval()
        for p in vae.parameters():
            p.requires_grad = False

    # -------------------------
    # Model
    # -------------------------
    if space == "latent":
        net_model = UNetModelWrapper(
            dim=(4, 32, 32),
            num_res_blocks=FLAGS.num_res_blocks,
            num_channels=FLAGS.num_channel,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions=parse_attention_resolutions(FLAGS.attention_resolutions),
            dropout=0.1,
        ).to(rank)
    else:
        net_model = UNetModelWrapper(
            dim=(3, FLAGS.image_size, FLAGS.image_size),
            num_res_blocks=FLAGS.pixel_num_res_blocks,
            num_channels=FLAGS.pixel_num_channel,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions=parse_attention_resolutions(FLAGS.pixel_attention_resolutions),
            dropout=0.1,
        ).to(rank)

    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)

    if FLAGS.parallel:
        net_model = DistributedDataParallel(net_model, device_ids=[rank])
        ema_model = DistributedDataParallel(ema_model, device_ids=[rank])

    # show model size
    if is_main:
        model_size = 0
        for param in net_model.parameters():
            model_size += param.data.nelement()
        print("Model params: %.2f M" % (model_size / 1024 / 1024))

    # -------------------------
    # Flow Matcher
    # -------------------------
    FM = build_flow_matcher()

    # -------------------------
    # Save dir
    # -------------------------
    savedir = os.path.join(
        FLAGS.output_dir,
        f"{FLAGS.model}_{space}_singlegpu" if not FLAGS.parallel else f"{FLAGS.model}_{space}_ddp{total_num_gpus}gpu",
    )
    if is_main:
        os.makedirs(savedir, exist_ok=True)

    # Fixed test batch for previews (main process only)
    if is_main:
        fixed_testA = next(iter(testA_loader)).to(rank)

    # -------------------------
    # Training loop
    # -------------------------
    # Calculate epochs from steps (for DDP sampler epoch setting)
    min_dataset_len = min(len(trainA), len(trainB))
    steps_per_epoch = math.ceil(min_dataset_len / batch_size_per_gpu)

    global_step = 0

    with trange(FLAGS.total_steps, dynamic_ncols=True, disable=not is_main) as pbar:
        epoch = 0
        while global_step < FLAGS.total_steps:
            # Set epoch for distributed samplers
            if samplerA is not None:
                samplerA.set_epoch(epoch)
            if samplerB is not None:
                samplerB.set_epoch(epoch)

            for _ in range(steps_per_epoch):
                if global_step >= FLAGS.total_steps:
                    break

                net_model.train()
                optim.zero_grad(set_to_none=True)

                xA = next(loopA).to(rank)
                xB = next(loopB).to(rank)

                if space == "latent":
                    with torch.no_grad():
                        zA = vae.encode(xA).latent_dist.mean
                        zB = vae.encode(xB).latent_dist.mean

                    t, zt, ut = FM.sample_location_and_conditional_flow(zA, zB)
                    vt = net_model(t, zt)
                    loss = torch.mean((vt - ut) ** 2)
                else:
                    t, xt, ut = FM.sample_location_and_conditional_flow(xA, xB)
                    vt = net_model(t, xt)
                    loss = torch.mean((vt - ut) ** 2)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), FLAGS.grad_clip)
                optim.step()
                sched.step()
                ema(net_model, ema_model, FLAGS.ema_decay)

                # Logging & saving (main process only)
                if is_main:
                    pbar.set_postfix(loss=float(loss.item()))
                    pbar.update(1)

                    if FLAGS.save_step > 0 and global_step % FLAGS.save_step == 0:
                        # For preview: need to unwrap DDP model
                        preview_net = net_model.module if FLAGS.parallel else net_model
                        preview_ema = ema_model.module if FLAGS.parallel else ema_model

                        translate_and_save_grid(
                            model=preview_net,
                            vae=vae,
                            xA=fixed_testA,
                            out_path=os.path.join(savedir, f"normal_translate_step_{global_step}.png"),
                        )
                        translate_and_save_grid(
                            model=preview_ema,
                            vae=vae,
                            xA=fixed_testA,
                            out_path=os.path.join(savedir, f"ema_translate_step_{global_step}.png"),
                        )

                        ckpt = {
                            "net_model": net_model.state_dict(),
                            "ema_model": ema_model.state_dict(),
                            "sched": sched.state_dict(),
                            "optim": optim.state_dict(),
                            "step": global_step,
                            "cfg": {
                                "model": FLAGS.model,
                                "space": space,
                                "data_root": FLAGS.data_root,
                                "sigma": FLAGS.sigma,
                            },
                        }
                        if space == "latent":
                            ckpt["cfg"]["vae_name"] = FLAGS.vae_name

                        torch.save(
                            ckpt,
                            os.path.join(savedir, f"{FLAGS.model}_{space}_step_{global_step}.pt"),
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
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        train(rank=device, total_num_gpus=total_num_gpus, argv=argv)


if __name__ == "__main__":
    app.run(main)