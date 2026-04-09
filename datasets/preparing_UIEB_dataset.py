"""
Resize paired images from two folders to 256x256 by:
- preserving aspect ratio
- using the same transformation for both images in each pair
- padding to 256x256 with reflection padding

Default input folders:
  raw-890
  reference-890

Inside the output root, images are saved using the same subfolder names:
  raw-890
  reference-890

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_OUT_ROOT = Path("/path/to/UIEB_dataset")


def list_image_files(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS],
        key=lambda p: p.name,
    )


def compute_resize_and_padding(width: int, height: int, target_size: int) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
    """Return ((new_width, new_height), (pad_left, pad_top, pad_right, pad_bottom))."""
    scale = min(target_size / width, target_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2
    pad_right = target_size - new_width - pad_left
    pad_bottom = target_size - new_height - pad_top

    return (new_width, new_height), (pad_left, pad_top, pad_right, pad_bottom)


def resize_with_reflect_padding(img: Image.Image, new_size: Tuple[int, int], padding: Tuple[int, int, int, int]) -> Image.Image:
    """Resize with bicubic interpolation, then reflect-pad to the target size."""
    img = img.convert("RGB")
    img = img.resize(new_size, Image.BICUBIC)

    arr = np.array(img)
    pad_left, pad_top, pad_right, pad_bottom = padding

    if any(v < 0 for v in padding):
        raise ValueError(f"Negative padding encountered: {padding}")

    if pad_left == pad_top == pad_right == pad_bottom == 0:
        return Image.fromarray(arr)

    arr = np.pad(
        arr,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="reflect",
    )
    return Image.fromarray(arr)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_pairs(raw_dir: Path, ref_dir: Path, out_root: Path, target_size: int) -> None:
    out_raw = out_root / raw_dir.name
    out_ref = out_root / ref_dir.name

    ensure_dir(out_raw)
    ensure_dir(out_ref)

    raw_files = {p.name: p for p in list_image_files(raw_dir)}
    ref_files = {p.name: p for p in list_image_files(ref_dir)}

    common_names = sorted(set(raw_files.keys()) & set(ref_files.keys()))
    raw_only = sorted(set(raw_files.keys()) - set(ref_files.keys()))
    ref_only = sorted(set(ref_files.keys()) - set(raw_files.keys()))

    print(f"Found {len(raw_files)} images in {raw_dir}")
    print(f"Found {len(ref_files)} images in {ref_dir}")
    print(f"Matched pairs: {len(common_names)}")
    print(f"Output raw folder: {out_raw}")
    print(f"Output ref folder: {out_ref}")

    if raw_only:
        print(f"Warning: {len(raw_only)} image(s) exist only in {raw_dir} and will be skipped.")
    if ref_only:
        print(f"Warning: {len(ref_only)} image(s) exist only in {ref_dir} and will be skipped.")

    processed = 0
    skipped = 0

    for name in common_names:
        raw_path = raw_files[name]
        ref_path = ref_files[name]

        try:
            with Image.open(raw_path) as raw_img, Image.open(ref_path) as ref_img:
                raw_img = raw_img.convert("RGB")
                ref_img = ref_img.convert("RGB")

                raw_w, raw_h = raw_img.size
                ref_w, ref_h = ref_img.size

                if (raw_w, raw_h) != (ref_w, ref_h):
                    print(
                        f"Skipping {name}: pair size mismatch "
                        f"raw={raw_w}x{raw_h}, ref={ref_w}x{ref_h}"
                    )
                    skipped += 1
                    continue

                new_size, padding = compute_resize_and_padding(raw_w, raw_h, target_size)

                raw_out = resize_with_reflect_padding(raw_img, new_size, padding)
                ref_out = resize_with_reflect_padding(ref_img, new_size, padding)

                if raw_out.size != (target_size, target_size) or ref_out.size != (target_size, target_size):
                    raise RuntimeError(
                        f"Output size error for {name}: raw={raw_out.size}, ref={ref_out.size}"
                    )

                raw_out.save(out_raw / name)
                ref_out.save(out_ref / name)
                processed += 1

        except Exception as e:
            print(f"Error processing {name}: {e}")
            skipped += 1

    print("\nDone.")
    print(f"Processed pairs: {processed}")
    print(f"Skipped pairs:   {skipped}")
    print(f"Saved raw images to: {out_raw}")
    print(f"Saved ref images to: {out_ref}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize paired images to 256x256 with reflection padding.")
    parser.add_argument("--raw_dir", type=str, default="/path/to/raw-890", help="Path to raw/input image folder")
    parser.add_argument("--ref_dir", type=str, default="/path/to/reference-890", help="Path to reference/target image folder")
    parser.add_argument(
        "--out_root",
        type=str,
        default=str(DEFAULT_OUT_ROOT),
        help="Root output folder. The processed images will be saved inside this path using the same subfolder names.",
    )
    parser.add_argument("--size", type=int, default=256, help="Target image size (default: 256)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    ref_dir = Path(args.ref_dir)
    out_root = Path(args.out_root)

    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw folder not found or not a directory: {raw_dir}")
    if not ref_dir.exists() or not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference folder not found or not a directory: {ref_dir}")
    if args.size <= 0:
        raise ValueError("--size must be a positive integer")

    process_pairs(raw_dir, ref_dir, out_root, args.size)


if __name__ == "__main__":
    main()
