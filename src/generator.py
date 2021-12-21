from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet

from .utils.image import LabImageBatch
from .utils.utils import WelfordMeter
from .utils.checkpoint import *


__all__ = ["build_res_u_net", "pretrain_generator"]


def build_res_u_net(n_input=1, n_output=2, size=256, arch="resnet18", pretrained=True):
    """First pretraining step for generator -> Use a pretrained U-Net

    Args:
        n_input: Image input dimension
        n_output: Image output dimension
        size:
        arch:
        pretrained:

    Returns:
        (Pretrained) U-Net
    """
    arch_dict = dict(
        resnet18=resnet18,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(arch_dict[arch], pretrained=pretrained, n_in=n_input, cut=-2)
    gen_net = DynamicUnet(body, n_output, (size, size)).to(device)
    return gen_net


def pretrain_generator(gen_net: nn.Module, train_dl: DataLoader[LabImageBatch],
                       criterion=nn.L1Loss(), optimizer=None, epochs=20,
                       checkpoints=None):
    """Second pretraining step for generator"""

    checkpoint, cp_after_each, cp_overwrite = set_cp_args(checkpoints)

    if not optimizer:
        optimizer = optim.Adam(gen_net.parameters(), lr=1e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for e in range(epochs):
        loss_meter = WelfordMeter()
        for batch in tqdm(train_dl):
            preds = gen_net(batch.L.to(device))
            preds.masked_fill_(batch.pad_mask, batch.pad_fill_value)  # enforce zero loss at padded values
            loss = criterion(preds, batch.ab.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), len(batch))

        print(f"Epoch {e + 1}/{epochs}")
        print(f"L1 Loss: {loss_meter.mean:.5f} +- {loss_meter.std:.4f}")

        if checkpoint and (e+1) % cp_after_each == 0:
            path = secure_cp_path(checkpoint + f"_epoch_{e + 1:02d}")
            save_model(gen_net, path, cp_overwrite)
