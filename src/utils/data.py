from typing import Optional, Union
from numpy.typing import ArrayLike
from pathlib import Path

from PIL import Image
from fastai.data.external import untar_data, URLs

import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from .image import *


__all__ = ["URLs", "ColorizationDataset", "make_dataloader", "get_image_paths", "LabImageDataLoader"]


LabImageDataLoader = DataLoader[LabImageBatch]


class ColorizationDataset(Dataset):

    def __init__(self, paths: ArrayLike, split: Optional[str] = "train"):
        """
        A PyTorch Dataset for color images.

        Args:
            paths: A list containing all image paths.
            split: Determines the type of image transforms that will be applied.
        """

        assert split in ("train", "test", "val"), f"Invalid option '{split}'"

        transforms = [
            # Uncomment for significant speed up
            # T.Resize((256, 256), T.InterpolationMode.BICUBIC),  # ATTENTION: This skews/distorts the images!
        ]
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
        return LabImage(rgb_=img)

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def collate_fn(batch):
        elem = batch[0]
        if isinstance(elem, LabImage):
            return LabImageBatch(batch)
        # Fallback
        else:
            from torch.utils.data._utils.collate import default_collate  # noqa
            return default_collate(batch)


def make_dataloader(batch_size=16, n_workers=4, pin_memory=True, rng=None,
                    **kwargs) -> LabImageDataLoader:
    dataset = ColorizationDataset(**kwargs)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=n_workers,
                            collate_fn=dataset.collate_fn, pin_memory=pin_memory,
                            generator=rng, shuffle=bool(rng))
    return dataloader


def get_image_paths(url: str, choose_n=-1, test=0.2, seed=1234) \
        -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    # Download and untar data
    dataset_path = untar_data(url)
    dataset_path = Path(dataset_path) / "train_sample"
    # Grab all image file names
    paths = dataset_path.glob("*.jpg")
    paths = np.array(list(paths))

    # Create random generator
    np_rng = np.random.default_rng(seed=seed)

    if choose_n > 0:
        paths = np_rng.choice(paths, choose_n, replace=False)

    if not test:
        return paths

    assert 0 < test < 1, f"Invalid option {test}. Must be a valid percentage."

    n = len(paths)
    indices = np.arange(n)
    np_rng.shuffle(indices)
    split_idx = int(n * test)
    test_idx = indices[:split_idx]
    train_idx = indices[split_idx:]
    return paths[train_idx], paths[test_idx]