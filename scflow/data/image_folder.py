from pathlib import Path

import torch
from PIL import Image


_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class ImageFolderNoLabel(torch.utils.data.Dataset):
    _EXTS = _EXTS

    def __init__(self, folder, transform):
        self.folder = folder
        self.transform = transform
        self.files = sorted(
            p for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in self._EXTS
        )
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in: {folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)


class ImageFolderWithPath(torch.utils.data.Dataset):
    _EXTS = _EXTS

    def __init__(self, folder, transform):
        self.files = sorted(
            p for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in self._EXTS
        )
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {folder}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), str(path)
