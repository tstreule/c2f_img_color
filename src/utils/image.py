import time
import warnings
from pathlib import Path
from typing import Union

from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2lab, lab2rgb

import numpy as np
import torch

__all__ = ["LabImage", "LabImageBatch"]

_IMG_SIZE = 2  # for plt
Array = Union[list, np.ndarray, torch.Tensor, Image.Image]





class LabImage:

    def __init__(self, *, rgb_: Array = None, lab_: Array = None,
                 lab: Array = None, L: Array = None, ab: Array = None):
        """
        A Dataset optimized for storing and converting LAB Images.

        Args:
            *args: catches positional arguments (no effect)
            rgb_: RGB image array of shape (n, m, 3)
            lab_: LAB image array of shape (n, m, 3)
            lab: LAB image array of shape (3, n, m)
            L: "L"-part of LAB image array of shape (1, n, m)
            ab: "ab"-part of LAB image array of shape (2, n, m)
        """
        assert not all(x is not None for x in [rgb_, lab_, lab, not (L is None or ab is None)]), \
            "Setting values in more than one way is ambiguous"

        self._lab = np.array([])  # 3 x n x m
        if any(x is not None for x in [rgb_, lab_]):
            self.from_true_values(rgb_=rgb_, lab_=lab_)
        elif any(x is not None for x in [lab, L, ab]):
            self.from_normalized_values(lab=lab, L=L, ab=ab)

    def __str__(self):
        cname = self.__class__.__name__
        return f"<{cname} lab={self._lab.shape}>"

    # === Data Setter ===

    def _store_lab(self, lab: np.ndarray, clip=True, tol=0.0):
        is_valid = np.greater_equal(lab, - 1.0 - tol) & np.less_equal(lab, + 1.0 + tol)
        num_violations = lab.size - is_valid.sum()
        if num_violations > 0:
            warnings.warn(
                f"Data appears to be non-normalized. {num_violations} / {lab.size} "
                f"values are out of the tolerance area: |x| <= 1.0 + {tol}.")
        if clip:
            self._lab = np.clip(lab, -1., 1.)
        else:
            self._lab = np.array(lab)

    def from_true_values(self, *, rgb_=None, lab_=None):
        assert not all(x is not None for x in [rgb_, lab_]), \
            "Setting both, rgb_ and lab_ values, is ambiguous"
        if rgb_ is not None:
            rgb_ = np.array(rgb_)
            lab_ = rgb2lab(rgb_).astype(np.float32)
        elif lab_ is not None:
            lab_ = np.array(lab_)
        # Normalize (between -1 and +1)
        lab = lab_.transpose((2, 0, 1))
        lab[0] = lab[0] / 50.0 - 1.0
        lab[1:] /= 110.0
        self._store_lab(lab=lab)
        return self

    def from_normalized_values(self, *, L=None, ab=None, lab=None):
        assert not all(x is not None for x in [lab, L, ab]), \
            "Setting both, lab and L-ab values, is ambiguous"
        if L is not None and ab is not None:
            lab = np.concatenate([L, ab], axis=0)
        elif lab is not None:
            lab = np.array(lab)
        self._store_lab(lab)
        return self

    def save(self,fname):
        Image.fromarray(np.asarray((self.rgb_ * 255)).astype(np.uint8)).save(fname)

    # === Data Getter ===

    @property
    def shape(self):
        return self._lab.shape  # 3 x n x m

    @property
    def lab(self) -> torch.Tensor:
        """shape=(3, n, m)"""
        return torch.tensor(self._lab)

    @property
    def L(self) -> torch.Tensor:
        """shape=(1, n, m)"""
        return self.lab[[0], ...]

    @property
    def ab(self) -> torch.Tensor:
        """shape=(2, n, m)"""
        return self.lab[[1, 2], ...]

    @property
    def lab_(self) -> np.ndarray:
        """shape=(n, m, 3)"""
        L_ = (self.L + 1.0) * 50.0
        ab_ = self.ab * 110.0
        return np.concatenate([L_, ab_], axis=0).transpose((1, 2, 0))

    @property
    def rgb_(self) -> np.ndarray:
        """shape=(n, m, 3)"""
        with warnings.catch_warnings():
            # Ignore UserWarning: Color data out of range: Z < 0
            warnings.simplefilter("ignore", UserWarning)
            return lab2rgb(self.lab_)

    # === Other ===

    def visualize(self, other: "LabImage" = None, show=True, save=False, path=None, fname=None):
        n_imgs = 3 if other else 2
        # Make figure
        fig, axs = plt.subplots(1, n_imgs, figsize=(n_imgs * _IMG_SIZE, _IMG_SIZE))
        axs[0].imshow(self.L[0].numpy(), cmap="gray")
        axs[1].imshow(self.rgb_)
        if other:
            axs[2].imshow(other.rgb_)
        # Prettify
        [ax.axis("off") for ax in axs]
        fig.tight_layout()
        fname = f"{fname}_" if fname is not None else ""
        show_save_image(fig, show, save, path, f"{fname}img")


class LabImageBatch:

    def __init__(self, batch: list[LabImage] = None, lab: Array = None,
                 L: Array = None, ab: Array = None, pad_mask: Array = None):
        """
        A class capable of efficiently storing batched `LabImage`s.

        Args:
            batch: array/batch of `LabImage`s
            L: an array of shape (batch_size, 1, n, m) containing "L"-part of all LAB images
            ab: an array of shape (batch_size, 2, n, m) containing "ab"-part of all LAB images
            pad_mask: a mask of shape (batch_size, 3, n, m) that indicates padded values
        """
        assert not all([batch, L, ab]), "Setting all values is ambiguous"

        self._lab_batch = np.ma.array([])  # n_batches x 3 x n x m
        if batch is not None:
            self.from_batch(batch)
        elif pad_mask is not None:
            if lab is not None:
                self.from_lab(lab, pad_mask)
            elif L is not None and ab is not None:
                self.from_L_ab(L, ab, pad_mask)

    def __str__(self):
        cname = self.__class__.__name__
        return f"<{cname} lab_batch={self._lab_batch.shape}>"

    # === Data Setter ===

    def _store_lab_batch(self, lab_batch, pad_mask, fill_value=-1.0):
        pad_mask = np.broadcast_to(pad_mask, lab_batch.shape).astype("bool")
        self._lab_batch = np.ma.array(lab_batch, mask=pad_mask, fill_value=fill_value)

    def from_lab(self, lab, pad_mask):
        if isinstance(lab, torch.Tensor):
            lab = lab.detach().numpy()
        if isinstance(pad_mask, torch.Tensor):
            pad_mask = pad_mask.detach().numpy()
        self._store_lab_batch(lab, pad_mask)
        return self

    def from_L_ab(self, L, ab, pad_mask):
        if all(isinstance(x, torch.Tensor) for x in [L, ab]):
            L = L.detach().numpy()
            ab = ab.detach().numpy()
        lab_batch = np.concatenate([L, ab], axis=1)
        self.from_lab(lab_batch, pad_mask)
        return self

    def from_batch(self, batch):
        assert all(isinstance(img, LabImage) for img in batch)

        img_shapes = np.array([img.shape for img in batch])
        img_size_max = np.max(img_shapes, axis=0)

        lab_batch = np.empty(shape=(len(batch), *tuple(img_size_max)))
        lab_batch[:] = np.NaN
        for i, img in enumerate(batch):
            idx1, idx2 = tuple(img_shapes[i, 1:])
            lab_batch[i, :, :idx1, :idx2] = img.lab.numpy()
        self._store_lab_batch(lab_batch, np.isnan(lab_batch))
        return self

    # === Data Getter ===

    @property
    def lab(self) -> torch.Tensor:
        """shape=(batch_size, 3, n, m)"""
        # Masked values are filled with pad_fill_value
        return torch.tensor(np.ma.filled(self._lab_batch), dtype=torch.float32)

    @property
    def L(self) -> torch.Tensor:
        """shape=(batch_size, 1, n, m)"""
        return self.lab[:, :1]

    @property
    def ab(self) -> torch.Tensor:
        """shape=(batch_size, 2, n, m)"""
        return self.lab[:, 1:]

    @property
    def pad_mask(self) -> torch.BoolTensor:
        """shape=(batch_size, 1, n, m)"""
        np_mask = np.ma.getmask(self._lab_batch)
        np_mask = np_mask[:, :1, :, :]  # make shape broadcastable
        return torch.BoolTensor(np_mask)

    @property
    def pad_fill_value(self) -> float:
        return self._lab_batch.fill_value

    def __len__(self) -> int:
        return self._lab_batch.shape[0]

    def __getitem__(self, idx) -> LabImage:
        padded_lab = np.ma.filled(self._lab_batch[idx], fill_value=np.NaN)
        # Remove padding in both dimensions
        mask = ~ np.isnan(padded_lab).any(axis=0, keepdims=True)
        ax1 = mask.any(axis=2).sum()
        ax2 = mask.any(axis=1).sum()
        lab = padded_lab[:, :ax1, :ax2]
        return LabImage(lab=lab)

    # === Other ===

    def visualize(self, other: "LabImageBatch" = None, draw_n=6,
                  show=True, save=False, path=None, fname=None):
        n_rows = 3 if other else 2
        # Handle number of images to draw
        if not draw_n or not (0 < draw_n < len(self)):
            draw_n = len(self)
        # If single image
        if draw_n == 1:
            return self[0].visualize(other[0], show=show, save=save, path=path, fname=fname)
        # Make figure
        fig, axs = plt.subplots(n_rows, draw_n, figsize=(draw_n * _IMG_SIZE, n_rows * _IMG_SIZE))
        for i, ax_ in enumerate(axs.T):
            ax_[0].imshow(self[i].L[0].numpy(), cmap="gray")
            ax_[1].imshow(self[i].rgb_)
            if other:
                ax_[2].imshow(other[i].rgb_)
        # Prettify
        [[ax.axis("off") for ax in ax_] for ax_ in axs]
        fig.tight_layout()
        fname = f"{fname}_" if fname is not None else ""
        show_save_image(fig, show, save, path, f"{fname}batch")


def show_save_image(fig, show: bool, save: bool, path=None, fname=None):
    if show:
        fig.show()
    if save:
        path = Path("imgs") if not path else Path(path)
        path.mkdir(parents=True, exist_ok=True)
        fname = f"{fname.rstrip('_')}_{time.time()}.png"
        fig.savefig(path / fname)

def load_image(path):
    img = Image.open(path).convert("RGB")
    return LabImage(rgb_=img)