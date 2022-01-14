from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import OrderedDict
from typing import Union

import torch
from torch import nn
from torchvision.utils import make_grid
from pytorch_lightning import LightningModule
from kornia.losses import ssim_loss, psnr_loss

from .discriminator import PatchDiscriminator
from .generator import build_res_u_net
from .utils.data import ColorizationBatch
from .utils.image import LabImage, LabImageBatch
from .utils.utils import *

import torchvision.transforms as T

__all__ = ["BaseModule", "PreTrainer", "ImageGAN", "C2FImageGAN"]

# TODO: handle learning rate according to batch size


def make_colorization_batch(data: Union[torch.Tensor, LabImage, LabImageBatch, ColorizationBatch]):
    if isinstance(data, torch.Tensor) and len(data.size()) == 3:
        data = LabImage(lab=data)
    if isinstance(data, torch.Tensor) and len(data.size()) == 4:
        data = LabImageBatch(lab=data, pad_mask=[0])
    if isinstance(data, LabImage):
        data = LabImageBatch(batch=[data])
    if isinstance(data, LabImageBatch):
        data = data.lab, (data.pad_mask, data.pad_fill_value)
    assert isinstance(data, tuple), f"Could not make `ColorizationBatch` out of data: {type(data)}"
    return data


def make_lab_image_batch(data):
    data = make_colorization_batch(data)
    data = LabImageBatch(lab=data[0], pad_mask=data[1][0])
    return data


class BaseModule(LightningModule, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.G_net: nn.Module
        self.D_net: nn.Module

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser):
        return parent_parser

    # === Losses ===

    @staticmethod
    def mae_criterion(imgs, pred_imgs):
        criterion = nn.L1Loss()
        return criterion(imgs, pred_imgs)

    @staticmethod
    def gan_criterion(imgs: torch.Tensor, pred_imgs):
        device = imgs.get_device() if imgs.is_cuda else "cpu"
        criterion = GANLoss("vanilla").to(device)
        return criterion(imgs, pred_imgs)

    # === Forward ===

    def forward(self, batch: ColorizationBatch) -> torch.Tensor:
        imgs, (pad_mask, pad_fill_value) = batch
        # Create input
        lights = imgs[:, :1]  # use `L` part
        dummy_colors = torch.zeros(imgs.shape[0], 2, *imgs.shape[2:]).type_as(imgs)
        pred_inputs = torch.cat([lights, dummy_colors], dim=1)
        pred_inputs = pred_inputs[:, :self.hparams.gen_net_params[0]]  # `L` or `Lab` part expected
        # Predict
        pred_imgs = torch.cat([lights, self.G_net(pred_inputs)], dim=1)
        pred_imgs.masked_fill_(pad_mask, pad_fill_value)  # enforce zero difference at padded values
        return pred_imgs

    def get_real_n_fake_imgs(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        real_imgs, _ = batch
        fake_imgs = self(batch)
        return real_imgs, fake_imgs

    # === Validation and testing

    def validation_step(self, batch, batch_idx):
        imgs, fake_imgs = self.get_real_n_fake_imgs(batch)
        self.log_losses(imgs, fake_imgs, batch_idx, "val")
        self.log_sample_images(imgs, fake_imgs, batch_idx=batch_idx)

    def test_step(self, batch, batch_idx):
        imgs, fake_imgs = self.get_real_n_fake_imgs(batch)
        self.log_losses(imgs, fake_imgs, batch_idx, "test")
        self.log_sample_images(imgs, fake_imgs, batch_idx=batch_idx)

    @abstractmethod
    def log_losses(self, real_imgs, fake_imgs, batch_idx, step=""):
        pass

    def log_sample_images(self, *imgs: torch.Tensor, batch_idx: int, max_n_imgs: int = 8):
        if batch_idx <= 4:
            sample_imgs = torch.cat([*imgs], dim=0)[:, :max_n_imgs]
            sample_grid = make_grid(sample_imgs, nrow=imgs[0].shape[0])
            # Transform color to rgb
            rgb = LabImage(lab=sample_grid.cpu()).rgb_.transpose((2, 0, 1))
            img_tensor = torch.tensor(rgb).type_as(imgs[0])
            self.logger.experiment.add_image("generated_images", img_tensor, self.current_epoch)

    @torch.no_grad()
    def colorize(self, data):
        batch = make_colorization_batch(data)
        batch = self(batch), batch[1]
        pred_batch = make_lab_image_batch(batch)
        return pred_batch

    @staticmethod
    def visualize(imgs, imgs2=None, **kwargs):
        imgs = make_lab_image_batch(imgs)
        imgs2 = None if imgs2 is None else make_lab_image_batch(imgs2)
        imgs.visualize(other=imgs2, **kwargs)


class PreTrainer(BaseModule):
    def __init__(
            self,
            gen_net_params: tuple[int],
            pretrain_lr: float,
            pretrain_betas: tuple[float] = (0.9, 0.99),
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[*kwargs.keys()])
        # Create generator
        self.G_net = init_weights(build_res_u_net(*gen_net_params))

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser):
        p = parent_parser.add_argument_group(f"{cls.__module__}.{cls.__qualname__}")
        p.add_argument("--gen_net_params", type=int,   default=(1, 2, 128), nargs=3)
        # Default values for optimizers, cf. https://arxiv.org/abs/1611.07004 section 3.3
        p.add_argument("--pretrain_lr",    type=float, default=1e-4, help="learning rate for generator")
        p.add_argument("--pretrain_betas", type=float, default=(0.9, 0.999), nargs=2,
                       help="(tuple) betas for generator")
        return parent_parser

    # === Training ===

    def configure_optimizers(self):
        lr = self.hparams.pretrain_lr
        betas = self.hparams.pretrain_betas
        optimizer = torch.optim.Adam(self.G_net.parameters(), lr=lr, betas=betas)
        return optimizer

    def training_step(self, batch, batch_idx):
        imgs, fake_imgs = self.get_real_n_fake_imgs(batch)
        # Get loss
        loss = self.mae_criterion(imgs[:, 1:], fake_imgs[:, 1:])  # use `ab` part only
        # Logging
        tqdm_dict = {"mae_loss": loss.detach()}
        log_dict = {"batch_size": len(imgs), **tqdm_dict}
        output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict, "log": log_dict})
        return output

    # === Validation and testing ===

    def log_losses(self, real_imgs, fake_imgs, batch_idx, step=""):
        # Get losses
        loss = self.mae_criterion(real_imgs[:, 1:], fake_imgs[:, 1:])  # use `ab` part only
        ssim = ssim_loss(real_imgs, fake_imgs, 5)
        psnr = psnr_loss(real_imgs, fake_imgs, 1.)
        # Log losses
        log_dict = {"mae_loss": loss, "ssim_loss": ssim, "psnr_loss": psnr}
        log_dict = {f"{step}_{name}": value for name, value in log_dict.items()}
        self.log_dict(log_dict, batch_size=len(real_imgs), sync_dist=True)


class ImageGAN(BaseModule):
    def __init__(
            self,
            pretrained_ckpt_path: str,
            gen_net_params: tuple[int],
            dis_net_params: tuple[int],
            gen_lr: float,
            dis_lr: float,
            gen_betas: tuple[float],
            dis_betas: tuple[float],
            gen_lambda_mae: float,
            **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[*kwargs.keys()])

        # Load/create generator
        if pretrained_ckpt_path is not None:
            pretrained = PreTrainer.load_from_checkpoint(pretrained_ckpt_path)
            assert (*gen_net_params,) == (*pretrained.hparams.gen_net_params,), (
                f"expected gen_net_params={pretrained.hparams.gen_net_params} "
                f"but found {gen_net_params} instead.")
            self.G_net = pretrained.G_net
        else:
            self.G_net = init_weights(build_res_u_net(*gen_net_params))

        # Create discriminator
        self.D_net = init_weights(PatchDiscriminator(*dis_net_params))

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser, *, return_group: bool = True):
        p = parent_parser.add_argument_group(f"{cls.__module__}.{cls.__qualname__}")
        p.add_argument("--gen_net_params", type=int,   default=(1, 2, 128), nargs=3)
        p.add_argument("--dis_net_params", type=int,   default=(3, 64, 3), nargs=3)
        # Default values for optimizers, cf. https://arxiv.org/abs/1611.07004 section 3.3
        p.add_argument("--gen_lr",         type=float, default=2e-4, help="learning rate for generator")
        p.add_argument("--dis_lr",         type=float, default=2e-4, help="learning rate for discriminator")
        p.add_argument("--gen_betas",      type=float, default=(.5, .999), nargs=2, help="(tuple) betas for generator")
        p.add_argument("--dis_betas",      type=float, default=(.5, .999), nargs=2, help="(tuple) betas for discriminator")
        p.add_argument("--gen_lambda_mae", type=float, default=100.0, help="weight factor for L1 loss")
        if return_group:
            return p
        return parent_parser

    # === Training ===

    def configure_optimizers(self):
        gen_lr = self.hparams.gen_lr
        gen_betas = self.hparams.gen_betas
        dis_lr = self.hparams.dis_lr
        dis_betas = self.hparams.dis_betas
        # Get optimizers
        opt_g = torch.optim.Adam(self.G_net.parameters(), lr=gen_lr, betas=gen_betas)
        opt_d = torch.optim.Adam(self.D_net.parameters(), lr=dis_lr, betas=dis_betas)
        return opt_d, opt_g  # note that the discriminator comes first

    def training_step(self, batch, batch_idx, optimizer_idx):
        imgs, fake_imgs = self.get_real_n_fake_imgs(batch)

        # --- Update discriminator ---
        if optimizer_idx == 0:
            set_requires_grad(self.D_net, True)

            # Get losses
            real_loss = self.gan_criterion(self.D_net(imgs), True)
            fake_loss = self.gan_criterion(self.D_net(fake_imgs.detach()), False)
            d_loss = (real_loss + fake_loss) / 2

            tqdm_dict = {"d_loss": d_loss.detach()}
            log_dict = {"batch_size": len(imgs), **tqdm_dict}
            output = OrderedDict({"loss": d_loss, "progress_bar": tqdm_dict, "log": log_dict})
            return output

        # --- Update generator ---
        if optimizer_idx == 1:
            set_requires_grad(self.D_net, False)

            # Get losses
            gan_loss = self.gan_criterion(self.D_net(fake_imgs), False)
            mae_loss = self.mae_criterion(imgs[:, 1:], fake_imgs[:, 1:])  # use `ab` part only
            g_loss = gan_loss + mae_loss * self.hparams.gen_lambda_mae

            tqdm_dict = {"g_loss": g_loss.detach()}
            log_dict = {"batch_size": len(imgs), **tqdm_dict}
            output = OrderedDict({"loss": g_loss, "progress_bar": tqdm_dict, "log": log_dict})
            return output

    # === Validation and testing ===

    def log_losses(self, real_imgs, fake_imgs, batch_idx, step=""):
        # Get losses
        gan_loss = self.gan_criterion(self.D_net(fake_imgs), False)
        mae_loss = self.mae_criterion(real_imgs[:, 1:], fake_imgs[:, 1:])  # use `ab` part only
        g_loss = gan_loss + mae_loss * self.hparams.gen_lambda_mae
        ssim = ssim_loss(real_imgs, fake_imgs, 5)
        psnr = psnr_loss(real_imgs, fake_imgs, 1.)
        # Log losses
        log_dict = {"g_loss": g_loss, "gan_loss": gan_loss, "mae_loss": mae_loss,
                    "ssim_loss": ssim, "psnr_loss": psnr}
        log_dict = {f"{step}_{name}": value for name, value in log_dict.items()}
        self.log_dict(log_dict, batch_size=len(real_imgs), sync_dist=True)


class C2FImageGAN(ImageGAN):
    def __init__(
            self,
            *args,
            shrink_size: float,
            min_ax_size: int,
            max_c2f_depth: int,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(*args, shrink_size, min_ax_size, max_c2f_depth)

    @classmethod
    def add_model_specific_args(cls, parent_parser: ArgumentParser, *args):
        p = super().add_model_specific_args(parent_parser, return_group=True)
        p.title = f"{cls.__module__}.{cls.__qualname__}"
        p.set_defaults(gen_net_params=(3, 2, 128))
        p.add_argument("--shrink_size", type=float, default=2, help="shrink size parameter"
                                                                    "(larger values speed up training and prediction)")
        p.add_argument("--min_ax_size", type=int, default=64, help="minimal size an image axis must have for recursion")
        p.add_argument("--max_c2f_depth", type=int, default=5, help="maximal recursive (c2f) depth")
        return parent_parser

    # === Forward ===

    def forward(self, batch, *, rec_depth=0):
        real_imgs, (pad_mask, pad_fill_value) = batch

        real_sizes = real_imgs.shape[2:]
        smaller_sizes = [int(size / self.hparams.shrink_size) for size in real_sizes]

        if any(size < self.hparams.min_ax_size for size in smaller_sizes) \
                or rec_depth >= self.hparams.max_c2f_depth:
            # Initialize dummy predictions
            # when not training `real_imgs` can also be just a "L"
            prev_pred_imgs = torch.zeros(real_imgs.shape[0], 3, *real_imgs.shape[2:]).type_as(real_imgs)
        else:
            resize = T.Resize(tuple(smaller_sizes), T.InterpolationMode.BICUBIC)
            resized_batch = resize(real_imgs), (resize(pad_mask), pad_fill_value)
            prev_pred_imgs = self(resized_batch, rec_depth=rec_depth+1)
            del resized_batch

        # Resizer for scaling up or down
        resize = T.Resize(tuple(real_sizes), T.InterpolationMode.BICUBIC)

        # Prediction
        lights = real_imgs[:, :1]  # use `L` part
        colors = resize(prev_pred_imgs[:, 1:])  # use `ab` part  # TODO: `.detach()`?
        pred_inputs = torch.cat([lights, colors], dim=1)
        del colors
        pred_imgs = torch.cat([lights, self.G_net(pred_inputs)], dim=1)
        del lights, pred_inputs
        pred_imgs.masked_fill_(resize(pad_mask), pad_fill_value)  # enforce zero difference at padded values

        return pred_imgs
