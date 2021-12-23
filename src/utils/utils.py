from typing import Union
from pathlib import Path

import numpy as np
import torch
from torch import nn


__all__ = [
    "get_device", "init_weights", "set_requires_grad", "GANLoss",  # PyTorch model utils
    "set_cp_args", "secure_cp_path",  # Checkpointing
    "WelfordMeter",  # Other
]


# === PyTorch model utils ===

def get_device(device: Union[str, torch.device] = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def init_weights(model: nn.Module, init="norm", gain=0.02):
    """Initialize weights with mean=0.0 and std=0.02 as proposed by
    https://arxiv.org/abs/1611.07004 in section 6.2
    """

    def init_fc(m: nn.Module):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and "Conv" in classname:
            if init == "norm":
                nn.init.normal_(m.weight.data, mean=0.0, std=gain)
            elif init == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif "BatchNorm2d" in classname:
            nn.init.normal_(m.weight.data, 1., gain)
            nn.init.constant_(m.bias.data, 0.)

    model.apply(init_fc)
    print(f"Model {model.__class__} initialized with {init} initialization")
    return model


def set_requires_grad(model: nn.Module, requires_grad: bool):
    for p in model.parameters():
        p.requires_grad = requires_grad


class GANLoss(nn.Module):
    def __init__(self, gan_mode='vanilla', real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        if gan_mode == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.criterion = nn.MSELoss()

    def get_labels(self, preds, target_is_real_img):
        labels = self.real_label if target_is_real_img \
            else self.fake_label
        return labels.expand_as(preds)

    def __call__(self, preds, target_is_real_img):
        labels = self.get_labels(preds, target_is_real_img)
        loss = self.criterion(preds, labels)
        return loss


# === Checkpointing ===

def set_cp_args(checkpoint=None):
    """
    Returns:
            checkpoint_args (tuple):
            - (str) checkpoint path
            - (int) checkpoint after each
            - (bool) overwrite checkpoint if already exists
    """
    if isinstance(checkpoint, str):
        checkpoint = (checkpoint, 20, True)

    path, after_each, overwrite = [fc(cp) for cp, fc in zip(checkpoint, [str, int, bool])]

    return path, after_each, overwrite


def secure_cp_path(name: str) -> str:
    path = str(name).strip(".pt") + ".pt"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


# === Other utils ===

class WelfordMeter:
    """
    Implements Welford's online algorithm for running averages and standard deviations.

    - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    - https://www.kite.com/python/answers/how-to-find-a-running-standard-deviation-in-python
    """

    def __init__(self):
        self.counter = 0
        self._M = 0
        self._S = 0
        self._main_buffer: list[np.ndarray] = []
        self._sub_buffer = None
        self._sub_buff_counter = 0
        self.reset()

    def reset(self):
        # Keep buffer as is but reset running averages and stds
        self.counter = 0
        self._M = 0
        self._S = 0
        # Add previous sub buffer (if exists) to main buffer
        if self._sub_buffer is not None and (~np.isnan(self._sub_buffer)).any():
            mask = np.isnan(self._sub_buffer).any(axis=1)
            self._main_buffer += [self._sub_buffer[~mask]]
        # Create new, empty sub buffer
        self._sub_buffer = np.empty((2, 3), dtype=np.float16)
        self._sub_buffer[:] = np.NaN
        self._sub_buff_counter = 0

    def update(self, val, count=1):
        self.counter += count

        delta1 = val - self._M
        self._M += count * delta1 / self.counter
        delta2 = val - self._M
        self._S += count * delta1 * delta2

        self._update_buffer(val)

    def _update_buffer(self, val):
        # Double buffer size if necessary
        buff_size = len(self._sub_buffer)
        if buff_size < self._sub_buff_counter + 1:
            new_buffer = np.empty((2 * buff_size, 3), dtype=np.float16)
            new_buffer[:] = np.NaN
            new_buffer[:buff_size] = self._sub_buffer
            self._sub_buffer = new_buffer
        # Write values
        self._sub_buffer[self._sub_buff_counter] = (val, self.mean, self.std)
        self._sub_buff_counter += 1

    @property
    def mean(self):
        return self._M

    @property
    def std(self):
        if self.counter == 1:
            return 0.0
        return np.sqrt(self._S / self.counter)

    @property
    def buffer_data(self):
        return self._main_buffer
