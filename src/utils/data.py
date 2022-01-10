from typing import Optional
from numpy.typing import ArrayLike
from pathlib import Path
import re

from PIL import Image
from fastai.data.external import untar_data, URLs
import multiprocessing

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

from .image import *

__all__ = ["URLs", "LabImageDataLoader", "ColorizationBatch",
           "ColorizationDataset", "ColorizationDataModule", "make_dataloader"]

LabImageDataLoader = DataLoader[LabImageBatch]
ColorizationBatch = tuple[torch.Tensor, tuple[torch.BoolTensor, float]]


def is_url(string: str) -> bool:
    pattern = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}" \
              r"\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
    match = re.match(pattern, string)
    return bool(match)


class ColorizationDataset(Dataset):

    def __init__(self, paths: ArrayLike, split: Optional[str] = "test", max_img_size=None):
        """
        A PyTorch Dataset for color images.

        Args:
            paths: A list containing all image paths.
            split: Determines the type of image transforms that will be applied.
            max_img_size: Limit image axes length. Mainly used for (significant) speed-up.
        """

        assert split in ("train", "test", "val"), f"Invalid option '{split}'"

        self.max_img_size = max_img_size

        transforms = []
        if split == "train":
            transforms.extend([
                T.RandomHorizontalFlip(),  # a little data augmentation!
            ])
        elif split in ("test", "val"):
            # (Currently) no additional image transforms
            pass

        self.transforms = T.Compose([*transforms])
        self.split = split
        self.paths = np.array(paths)

    def __getitem__(self, item):
        img = Image.open(self.paths[item]).convert("RGB")
        img = self.transforms(img)
        img = self._limit_size(img)
        return LabImage(rgb_=img)

    def __len__(self):
        return len(self.paths)

    def _limit_size(self, img):
        if self.max_img_size is not None:
            sizes = np.array(img.size, dtype=int)
            max_sizes = np.round(self.max_img_size / max(sizes) * sizes).astype(int)
            new_sizes = np.min([sizes, max_sizes], axis=0)
            resize = T.Resize(tuple([new_sizes[1], new_sizes[0]]), T.InterpolationMode.BICUBIC)
            return resize(img)
        else:
            return img

    @staticmethod
    def collate_fn(batch):
        elem = batch[0]
        if isinstance(elem, LabImage):
            batch = LabImageBatch(batch)
            return batch.lab, (batch.pad_mask, batch.pad_fill_value)
        # Fallback
        else:
            from torch.utils.data._utils.collate import default_collate  # noqa
            return default_collate(batch)


class ColorizationDataModule(LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int = 16,
            dataset_size: int = -1,
            max_img_size: int = None,
            tr_split: float = 0.7,
            vl_split: float = 0.15,
            te_split: float = 0.15,
            num_workers: int = None,
            seed: int = None,
    ):
        """

        Args:
            data_dir: path to local dataset or link to a `fastai` dataset
            batch_size: batch size of dataloader
            dataset_size: number of images to draw from dataset
            max_img_size: maximal image size dataset will yield (significant speed-up for small values!)
            tr_split: train split size
            vl_split: validation split size
            te_split: test split size
            num_workers: number of workers for dataloader
            seed: seed for dataset and dataloader
        """
        super().__init__()
        self.data_dir = self.handle_data_dir(data_dir)
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.dataset_kwargs = {"max_img_size": max_img_size}
        self.splits = self.handle_split_sizes(tr_split, vl_split, te_split)
        self.num_workers = self.handle_num_workers(num_workers)
        self.seed = seed
        # Other, not yet defined
        self.data_train = None
        self.data_val = None
        self.data_test = None

    @staticmethod
    def handle_data_dir(data_dir: str) -> Path:
        if is_url(data_dir):
            # Download
            return Path(untar_data(data_dir)) / "train_sample"
        else:
            return Path(data_dir)

    @staticmethod
    def handle_split_sizes(*args) -> list[float]:
        splits = [*args]
        return [float(s) / sum(splits) for s in splits]  # normalize

    @staticmethod
    def handle_num_workers(num_workers: int):

        if num_workers is not None:
            return int(num_workers)

        if torch.cuda.is_available():
            return torch.cuda.device_count()
        else:
            n_cpu = multiprocessing.cpu_count()
            return max(1, int(0.75 * n_cpu))

    def setup(self, stage: Optional[str] = None) -> None:
        # Grab all image file names
        paths = self.data_dir.glob("*.jpg")
        paths = np.array(list(paths))
        # Check dataset size limit
        if self.dataset_size < 0:
            self.dataset_size = len(paths)
        # Shuffle
        if self.seed is not None:
            indices = torch.randperm(len(paths), generator=self.torch_rng)
            paths = paths[indices]
        # Assign path subsets
        cut = [int(self.dataset_size * sum(self.splits[:i+1])) for i, _ in enumerate(self.splits)]
        train_paths = paths[0:cut[0]]
        val_paths = paths[cut[0]:cut[1]]
        test_paths = paths[cut[1]:self.dataset_size]
        del paths

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.data_train = ColorizationDataset(train_paths, split="train", **self.dataset_kwargs)
            self.data_val = ColorizationDataset(val_paths, split="val", **self.dataset_kwargs)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.data_test = ColorizationDataset(test_paths, split="test", **self.dataset_kwargs)

    @property
    def torch_rng(self):
        rng = torch.Generator()
        rng.manual_seed(self.seed)
        return rng

    @property
    def dl_kwargs(self):
        kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=ColorizationDataset.collate_fn,
        )
        return kwargs

    @property
    def dl_shuffle_kwargs(self):
        kwargs = {}
        if self.seed is not None:
            kwargs["shuffle"] = True
            kwargs["generator"] = self.torch_rng
        return kwargs

    def train_dataloader(self):
        return DataLoader(self.data_train, **self.dl_kwargs, **self.dl_shuffle_kwargs)

    def val_dataloader(self):
        return DataLoader(self.data_val, **self.dl_kwargs)

    def test_dataloader(self):
        return DataLoader(self.data_test, **self.dl_kwargs)

    def extract_batch_size(self):
        return self.batch_size


def make_dataloader(
        data_dir: str, batch_size=16, n_workers=1, pin_memory=True, rng=None, **kwargs
) -> LabImageDataLoader:
    paths = ColorizationDataModule.handle_data_dir(data_dir).glob("*.jpg")
    dataset = ColorizationDataset(list(paths), **kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            collate_fn=dataset.collate_fn, pin_memory=pin_memory,
                            generator=rng, shuffle=bool(rng))
    return dataloader
