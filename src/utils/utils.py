from typing import Union
from pathlib import Path

import numpy as np
import torch
from torch import nn


__all__ = [
    "init_weights", "set_requires_grad", "GANLoss",  # PyTorch model utils
]


# === PyTorch model utils ===

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
        return labels.type_as(preds).expand_as(preds)

    def __call__(self, preds, target_is_real_img):
        labels = self.get_labels(preds, target_is_real_img)
        loss = self.criterion(preds, labels)
        return loss
