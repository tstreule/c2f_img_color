from tqdm import tqdm
import warnings

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .generator import build_res_u_net
from .discriminator import PatchDiscriminator
from .utils.image import LabImageBatch
from .utils.utils import WelfordMeter
from .utils.checkpoint import *


__all__ = ["GANLoss", "ImageGAN"]


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


def init_model(model: nn.Module, device: torch.device):
    model = model.to(device)
    model = init_weights(model)
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


class ImageGAN:

    def __init__(self, gen_net: nn.Module = None, dis_net: nn.Module = None,
                 gen_opt_params: dict = None, dis_opt_params: dict = None,
                 gen_lambda_mae=100.0, device: torch.device = None):
        """
        Train agent for Image GAN.

        What's finally of interest is the trained `gen_net` for colorizing images.

        Args:
            gen_net: Generator network.
            dis_net: Discriminator network.
            gen_opt_params: Parameters for generator network optimizer.
            dis_opt_params: Parameters for discriminator network optimizer.
            gen_lambda_mae: Mixing parameter for generator loss.
                (loss = gan_loss + mae_loss * gen_lambda_mae)
            device: PyTorch device ("cpu" or "cuda").
        """
        super().__init__()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
            if device is None else torch.device(device)
        self._lambda_mae = gen_lambda_mae

        # Create generator and discriminator
        self.gen_net = init_model(build_res_u_net(1, 2), self._device) \
            if gen_net is None else gen_net.to(self._device)
        self.dis_net: PatchDiscriminator
        self.dis_net = init_model(PatchDiscriminator(3, 64, 3), self._device) \
            if dis_net is None else dis_net.to(self._device)

        # Define optimizer
        def opt_params(params: dict):
            if params is None:
                # Return default parameters as proposed by
                # https://arxiv.org/abs/1611.07004 section 3.3
                return dict(lr=2e-4, betas=(0.5, 0.999))
            else:
                return params
        self._gen_opt = optim.Adam(self.gen_net.parameters(), **opt_params(gen_opt_params))
        self._dis_opt = optim.Adam(self.dis_net.parameters(), **opt_params(dis_opt_params))

        # Define criteria
        self._gan_crit = GANLoss(gan_mode="vanilla").to(self._device)
        self._mae_crit = nn.L1Loss()

        # Logging
        self.epoch = 0
        self.loss_meters = self._create_loss_meters()

    # === Model updates ===

    def optimize(self, batch: LabImageBatch):
        L = batch.L.to(self._device)
        ab = batch.ab.to(self._device)
        real_imgs = torch.cat([L, ab], dim=1)
        fake_imgs = torch.cat([L, self.gen_net(L)], dim=1)
        # Overwrite padding of fake_imgs with real_imgs
        mask = batch.get_padding_mask()
        fake_imgs[mask] = -1

        # Update discriminator
        self.dis_net.train()
        set_requires_grad(self.dis_net, True)
        self._dis_opt.zero_grad()
        self._dis_loss(real_imgs, fake_imgs).backward()
        self._dis_opt.step()

        # Update generator
        self.gen_net.train()
        set_requires_grad(self.dis_net, False)
        self._gen_opt.zero_grad()
        self._gen_loss(real_imgs, fake_imgs).backward()
        self._gen_opt.step()

    def _dis_loss(self, real_imgs: torch.Tensor, fake_imgs: torch.Tensor) \
            -> torch.Tensor:
        real_preds = self.dis_net(real_imgs)
        fake_preds = self.dis_net(fake_imgs.detach())
        real_loss = self._gan_crit(real_preds, True)
        fake_loss = self._gan_crit(fake_preds, False)
        dis_loss = (real_loss + fake_loss) / 2.
        # Logging
        loss_values = dict(dis_loss_real=real_loss, dis_loss_fake=fake_loss, dis_loss=dis_loss)
        self.update_loss_meters(loss_values, count=real_imgs.size(0))
        return dis_loss

    def _gen_loss(self, real_imgs: torch.Tensor, fake_imgs: torch.Tensor) \
            -> torch.Tensor:
        fake_preds = self.dis_net(fake_imgs)
        gan_loss = self._gan_crit(fake_preds, False)
        mae_loss = self._mae_crit(real_imgs[:, 1:], fake_imgs[:, 1:])  # use `ab` part only
        gen_loss = gan_loss + mae_loss * self._lambda_mae
        # Logging
        loss_values = dict(gen_loss_gan=gan_loss, gen_loss_mae=mae_loss, gen_loss=gen_loss)
        self.update_loss_meters(loss_values, count=real_imgs.size(0))
        return gen_loss

    # === Logging ===

    @staticmethod
    def _create_loss_meters() -> dict[str, WelfordMeter]:
        loss_names = ["dis_loss_fake", "dis_loss_real", "dis_loss",  # TODO: What about 'SSIM' loss?
                      "gen_loss_gan", "gen_loss_mae", "gen_loss"]
        loss_meters = {name: WelfordMeter() for name in loss_names}
        return loss_meters

    def reset_loss_meters(self):
        for name, meter in self.loss_meters.items():
            meter.reset()

    def update_loss_meters(self, update_dict: dict[str, float], count=1):
        update_dict = {name: float(value)  # detach tensor properties
                       for name, value in update_dict.items()}
        for loss_name in update_dict:
            if loss_name in self.loss_meters:
                self.loss_meters[loss_name].update(update_dict[loss_name], count)
            else:
                print(f"Warning: Loss name {loss_name} not recognized.")

    def log_results(self):
        for loss_name, loss_meter in self.loss_meters.items():
            print(f"{loss_name}: {loss_meter.mean:.5f} +- {loss_meter.std:.4f}")

    # === Main training loop ===

    def train(self, train_dl: DataLoader[LabImageBatch], epochs=20, display_every=100,
              checkpoints=None):
        """
        Main training loop.

        Args:
            train_dl: Image data loader.
            epochs: Number of epochs for training.
            display_every: Log after `display_every` optimizing steps.
        """

        checkpoint, cp_after_each, cp_overwrite = set_cp_args(checkpoints)

        for e in range(epochs):
            if self.epoch > e:
                continue
            self.epoch = e

            self.reset_loss_meters()  # logging

            for i, batch in tqdm(enumerate(train_dl)):
                self.optimize(batch)

                if (i + 1) % display_every == 0:
                    print(f"\nEpoch {e+1}/{epochs}")
                    print(f"Iteration {i}/{len(train_dl)}")
                    self.log_results()
                    # Visualize generated images
                    self.gen_net.eval()
                    pred_imgs = LabImageBatch(L=batch.L, ab=self.gen_net(batch.L))
                    pred_imgs.visualize(other=batch, show=False, save=True)
                    self.gen_net.train()

            if checkpoint and (e + 1) % cp_after_each == 0:
                path = secure_cp_path(checkpoint + f"_epoch_{e+1:02d}")
                self.save_model(path, cp_overwrite)

    # === Save and load model ===

    def save_model(self, save_as: str, overwrite=True):
        save_dict = self.save_dict
        save_dict["torch"] = {name: getattr(self, attr_name).state_dict()
                              for name, attr_name in save_dict["torch"].items()}

        if overwrite:
            torch.save(save_dict, secure_cp_path(save_as))
        else:
            warnings.warn(f"Model {save_as} not saved. Overwriting prohibited.")

    def load_model(self, load_from: str):
        save_dict = self.save_dict
        checkpoint = torch.load(load_from)

        for name, attr_name in save_dict["torch"].items():
            getattr(self, attr_name).load_state_dict(checkpoint["torch"][name])
        for name, attr_name in save_dict["other"].items():
            setattr(self, attr_name, checkpoint["other"][name])

    @property
    def save_dict(self):
        return {
            "torch": dict(
                gen_model_state_dict="gen_net",
                gen_optim_state_dict="_gen_opt",
                dis_model_state_dict="dis_net",
                dis_optim_state_dict="_dis_opt",
            ),
            "other": dict(
                epoch="epoch",
                loss_meters="loss_meters",
            )
        }
