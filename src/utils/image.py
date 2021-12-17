from typing import Iterable
import time
import matplotlib.pyplot as plt

from skimage.color import rgb2lab, lab2rgb

import numpy as np
import torch
import torchvision.transforms as T


class LabImage:
    """
    Data container for L*a*b images which quick and easy can be converted
    to RGB and vice versa.

    Parameters
    ----------
    lab : array-like, optional
        L*a*b data array.
    rgb : array-like, optional
        RGB data array.
    L : array-like, optional
        L-part of L*a*b data array. Has no effect when `ab` parameter is None!
    ab : array-like, optional
        ab-part of L*a*b data array. Has no effect when `L` parameter is None!
    """

    def __init__(self, lab=None, rgb=None, L=None, ab=None):
        self._lab: np.ndarray
        self.from_lab(lab) if (lab is not None) else None
        self.from_rgb(rgb) if (rgb is not None) else None
        self.from_L_ab(L, ab) if (L is not None) and (ab is not None) else None

    # === Data Setter ===

    def _store(self, lab=None):
        self._lab = np.array(lab)
        return self

    def from_lab(self, lab):
        return self._store(lab=lab)

    def from_rgb(self, rgb):
        rgb = np.array(rgb)
        lab = rgb2lab(rgb).astype("float32")
        return self._store(lab=lab)

    def from_L_ab(self, L, ab):
        L = (L + 1.0) * 50.0
        ab = ab * 110.0
        lab = np.concatenate([L, ab], axis=-1)
        return self.from_lab(lab)

    # === Data Getter ===

    @property
    def rgb(self):
        rgb = lab2rgb(self._lab)
        return torch.tensor(rgb)

    @property
    def lab(self):
        return torch.tensor(self._lab)

    @property
    def L(self):
        return self.lab[:, :, 0].unsqueeze(-1) / 50. - 1.  # between -1 and 1

    @property
    def ab(self):
        return self.lab[:, :, 1:] / 110.  # between -1 and 1

    # === Other ===

    def visualize(self, save=False):
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(self.L.numpy(), cmap="gray")
        axs[1].imshow(self.rgb.numpy())
        [ax.axis("off") for ax in axs]
        fig.tight_layout()
        fig.show()
        if save:
            fig.savefig(f"colorization_{time.time()}.png")


class LabImageBatch:
    """
    Data batch container for L*a*b images which quick and easy can be
    converted to RGB and vice versa.

    Parameters
    ----------
    batch : Iterable[LabImage]
    """

    def __init__(self, batch):
        self.batch: list[LabImage] = []
        self._collate(batch)

    def _collate(self, batch, size_criterion=np.max):
        assert all([isinstance(elem, LabImage) for elem in batch])

        # Create image cropper s.t. all images in batch have same size
        img_sizes = np.array([elem.rgb.shape[:2] for elem in batch])
        size = size_criterion(img_sizes, axis=0).astype("int")
        cropper = T.RandomCrop(size=size, pad_if_needed=True)

        # Store cropped (or padded) images and return
        batch = [LabImage(lab=cropper(img.lab.permute(2, 0, 1)).permute(1, 2, 0))
                 for img in batch]
        self.batch = batch

        return self

    # === Data Getter ===

    def __getitem__(self, item) -> LabImage:
        return self.batch[item]

    @property
    def lab(self):
        labs = [img.lab for img in self.batch]  # collate
        return torch.cat(labs, dim=0)

    @property
    def rgb(self):
        rgbs = [img.rgb for img in self.batch]  # collate
        return torch.cat(rgbs, dim=0)

    @property
    def L(self):
        Ls = [img.L for img in self.batch]  # collate
        return torch.cat(Ls, dim=0)

    @property
    def ab(self):
        abs_ = [img.ab for img in self.batch]  # collate
        return torch.cat(abs_, dim=0)

    @property
    def batch_size(self):
        return len(self.batch)

    # === Other ===

    def visualize(self, draw_n=None, save=False):
        if not draw_n or not (0 < draw_n < self.batch_size):
            draw_n = self.batch_size
        fig, axs = plt.subplots(2, draw_n, figsize=(3*draw_n, 6))
        for i, ax_ in enumerate(axs.T):
            ax_[0].imshow(self[i].L.numpy(), cmap="gray")
            ax_[1].imshow(self[i].rgb.numpy())
            [ax.axis("off") for ax in ax_]
        fig.tight_layout()
        fig.show()
        if save:
            fig.savefig(f"colorization_{time.time()}.png")
