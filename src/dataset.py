import os
import tarfile
from pathlib import Path
from typing import Optional

import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

from configs.default import TrainConfig


DATASET_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/"
IMAGES_TAR = "images.tar"
LISTS_TAR = "lists.tar"


def download_and_extract(data_dir: Path):
    """Download Stanford Dogs dataset if not present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "Images"
    lists_dir = data_dir

    if images_dir.exists() and (lists_dir / "train_list.mat").exists():
        return

    import urllib.request

    for fname in [IMAGES_TAR, LISTS_TAR]:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(DATASET_URL + fname, fpath)
        print(f"Extracting {fname}...")
        with tarfile.open(fpath, "r") as tar:
            tar.extractall(data_dir)

    print("Dataset ready.")


class StanfordDogsDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / "Images"
        self.transform = transform

        mat_file = "train_list.mat" if split == "train" else "test_list.mat"
        mat_path = self.data_dir / mat_file
        mat = scipy.io.loadmat(str(mat_path))

        self.file_list = [str(item[0][0]) for item in mat["file_list"]]
        self.labels = [int(item[0]) - 1 for item in mat["labels"]]

        breed_dirs = sorted(os.listdir(self.images_dir))
        self.class_names = [d.split("-", 1)[1] for d in breed_dirs]

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        img_path = self.images_dir / self.file_list[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def create_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test DataLoaders with proper val split."""
    download_and_extract(cfg.data_dir)

    train_dataset = StanfordDogsDataset(cfg.data_dir, split="train", transform=get_train_transform(cfg.image_size))
    val_dataset = StanfordDogsDataset(cfg.data_dir, split="train", transform=get_val_transform(cfg.image_size))
    test_dataset = StanfordDogsDataset(cfg.data_dir, split="test", transform=get_val_transform(cfg.image_size))

    # Split train into train/val using stratified indices
    n = len(train_dataset)
    generator = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(n, generator=generator).tolist()
    val_size = int(n * cfg.val_split)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=True if cfg.num_workers > 0 else False,
    )

    train_loader = DataLoader(train_subset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
