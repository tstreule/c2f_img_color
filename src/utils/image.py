from numpy.typing import ArrayLike
import time
import warnings
import matplotlib.pyplot as plt

from skimage.color import rgb2lab, lab2rgb

import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import crop as T_functional_crop


__all__ = ["LabImage", "LabImageBatch"]


class LabImage:

    def __init__(self, lab: ArrayLike = None, rgb: ArrayLike = None,
                 L: ArrayLike = None, ab: ArrayLike = None):
        """
        Data container for L*a*b images which quick and easy can be converted
        to RGB and vice versa.

        Args:
            lab: L*a*b data array.
            rgb: RGB data array.
            L: `L`-part of L*a*b data array.
                Has no effect when `ab` parameter is None!
            ab: `ab`-part of L*a*b data array.
                Has no effect when `L` parameter is None!
        """

        self._lab: np.ndarray
        self.from_lab(lab) if (lab is not None) else None
        self.from_rgb(rgb) if (rgb is not None) else None
        self.from_L_ab(L, ab) if (L is not None) and (ab is not None) else None

    # === Data Setter ===

    def _store(self, lab=None):
        self._lab = np.array(lab)
        return self

    def from_lab(self, lab):
        lab = np.array(lab)
        if len(lab) == 3:  # (3, n, n)
            lab = lab.transpose((1, 2, 0))
        return self._store(lab=lab)

    def from_rgb(self, rgb):
        rgb = np.array(rgb)
        lab = rgb2lab(rgb).astype("float32")
        return self._store(lab=lab)

    def from_L_ab(self, L, ab):
        L = (L + 1.0) * 50.0
        ab = ab * 110.0
        lab = np.concatenate([L, ab], axis=0)
        return self.from_lab(lab)

    # === Data Getter ===

    def __str__(self):
        cname = self.__class__.__name__
        return f"{cname}{self._lab.shape}"

    @property
    def rgb(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # warnings.filterwarnings("ignore", "UserWarning")
            # Ignore UserWarning: Color data out of range: Z < 0
            rgb = lab2rgb(self._lab)
        return torch.tensor(rgb)

    @property
    def lab(self):
        return torch.tensor(self._lab).permute(2, 0, 1)

    @property
    def L(self):
        return self.lab[[0], ...] / 50. - 1.  # between -1 and 1

    @property
    def ab(self):
        return self.lab[[1, 2], ...] / 110.  # between -1 and 1

    # === Other ===

    def visualize(self, save=False):
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))
        axs[0].imshow(self.L[0].numpy(), cmap="gray")
        axs[1].imshow(self.rgb.numpy())
        [ax.axis("off") for ax in axs]
        fig.tight_layout()
        fig.show()
        if save:
            fig.savefig(f"colorization_{time.time()}.png")


class LabImageBatch:

    def __init__(self, batch: list[LabImage] = None,
                 L: torch.Tensor = None, ab: torch.Tensor = None):
        """
        Data batch container for L*a*b images which quick and easy can be
        converted to RGB and vice versa.

        Args:
            batch: An iterable containing `LabImage`s
            L: 4D Tensor with `LabImage.L` attributes
            ab: 4D Tensor with `LabImage.ab` attributes
        """

        self.batch: list[LabImage] = []
        self.padding: list[list[int]] = []  # padding for the left, top, right and bottom borders respectively

        if batch is not None:
            self._collate(batch)
        elif (L is not None) and (ab is not None):
            self.from_L_ab(L, ab)

    def _collate(self, batch: list[LabImage], size_criterion: callable = np.max):
        assert all([isinstance(img, LabImage) for img in batch])

        # Create image cropper s.t. all images in batch have same size
        img_sizes = np.array([img.rgb.shape[:2] for img in batch])
        if size_criterion == np.max:
            size_diffs = - img_sizes + [size_criterion(img_sizes, axis=0)]
            self.padding = [[0, 0, right_pad, bottom_pad] for bottom_pad, right_pad in size_diffs]
            transforms = [T.Pad(padding=pad) for pad in self.padding]
        else:
            size = size_criterion(img_sizes, axis=0).astype("int")
            transforms = [T.RandomCrop(size=size, pad_if_needed=True)
                          for _ in range(len(batch))]

        # Store cropped (or padded) images and return
        batch = [LabImage(lab=t(img.lab)) for t, img in zip(transforms, batch)]
        self.batch = batch

        return self

    # === Data Setter ===

    def from_L_ab(self, L, ab):
        batch = [LabImage(L=L_.detach(), ab=ab_.detach()) for L_, ab_ in zip(L, ab)]
        self.batch = batch

    # === Data Getter ===

    def __getitem__(self, item) -> LabImage:
        image = self.batch[item]
        if self.padding:
            height, width = np.array(image.lab.shape[1:]) - self.padding[item][2:]
            image = LabImage(lab=T_functional_crop(image.lab, 0, 0, height, width))
        return image

    @property
    def lab(self):
        labs = [img.lab for img in self.batch]  # collate
        return torch.stack(labs, dim=0)

    @property
    def rgb(self):
        rgbs = [img.rgb for img in self.batch]  # collate
        return torch.stack(rgbs, dim=0)

    @property
    def L(self):
        Ls = [img.L for img in self.batch]  # collate
        return torch.stack(Ls, dim=0)

    @property
    def ab(self):
        abs_ = [img.ab for img in self.batch]  # collate
        return torch.stack(abs_, dim=0)

    @property
    def batch_size(self):
        return len(self.batch)

    # === Other ===

    def visualize(self, other=None, draw_n=None, save=False):
        other: LabImageBatch
        n_rows = 3 if other else 2

        if not draw_n or not (0 < draw_n < self.batch_size):
            draw_n = self.batch_size

        fig, axs = plt.subplots(n_rows, draw_n, figsize=(3*draw_n, 3*n_rows))
        for i, ax_ in enumerate(axs.T):
            ax_[0].imshow(self[i].L[0].numpy(), cmap="gray")
            ax_[1].imshow(self[i].rgb.numpy())
            if other:
                ax_[2].imshow(other[i].rgb.numpy())
            [ax.axis("off") for ax in ax_]
        fig.tight_layout()
        fig.show()
        if save:
            fig.savefig(f"colorization_{time.time()}.png")
