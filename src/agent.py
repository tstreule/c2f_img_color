import warnings
from pathlib import Path
from tqdm import tqdm
import time

import torch
from torch import nn, optim

from .discriminator import PatchDiscriminator
from .generator import build_res_u_net
from .utils.data import LabImageDataLoader
from .utils.image import LabImageBatch, LabImage
from .utils.utils import *

import torchvision.transforms as T
# === Loss Meters ===

LossMeterDict = dict[str, WelfordMeter]


def create_loss_meters(pretraining=False) -> LossMeterDict:
    if pretraining:
        loss_names = ["mae_loss"]
    else:
        # TODO: What about 'SSIM' loss?
        loss_names = ["dis_loss_fake", "dis_loss_real", "dis_loss",
                      "gen_loss_gan", "gen_loss_mae", "gen_loss"]
    loss_meters = {name: WelfordMeter() for name in loss_names}
    return loss_meters


def reset_loss_meters(loss_meters: LossMeterDict):
    for name, meter in loss_meters.items():
        meter.reset()


def update_loss_meters(loss_meters: LossMeterDict, update_dict: dict[str, float], count=1):
    update_dict = {name: float(value)  # detach tensor properties through float(...)
                   for name, value in update_dict.items()}
    for loss_name in update_dict:
        if loss_name in loss_meters:
            loss_meters[loss_name].update(update_dict[loss_name], count)
        else:
            warnings.warn(f"Warning: Loss name {loss_name} not recognized.")


def log_results(loss_meters: LossMeterDict):
    fill = max(len(name) for name in loss_meters)
    for loss_name, loss_meter in loss_meters.items():
        print(f"{loss_name: <{fill}}: {loss_meter.mean:.4f} +- {loss_meter.std:.4f}")


# === Main ===

class ImageGANAgent:

    def __init__(self, gen_net: nn.Module = None, dis_net: nn.Module = None,
                 gen_net_params=(1, 2, 128), dis_net_params=(3, 64, 3),
                 pre_opt_params: dict = None, gen_opt_params: dict = None,
                 dis_opt_params: dict = None, gen_lambda_mae=100.0, device=None):

        self._device = get_device(device)

        # Create generator and discriminator
        generator = init_weights(build_res_u_net(*gen_net_params)) if gen_net is None else gen_net
        discriminator = init_weights(PatchDiscriminator(*dis_net_params)) if dis_net is None else dis_net
        self.gen_net = generator.to(self._device)
        self.dis_net = discriminator.to(self._device)

        # Create model optimizer
        default_params = dict(lr=2e-4, betas=(0.5, 0.999))  # cf. https://arxiv.org/abs/1611.07004 section 3.3
        pre_opt_params = dict(lr=1e-4) if pre_opt_params is None else pre_opt_params
        gen_opt_params = default_params if gen_opt_params is None else gen_opt_params
        dis_opt_params = default_params if dis_opt_params is None else dis_opt_params
        self._pre_opt = optim.Adam(self.gen_net.parameters(), **pre_opt_params)
        self._gen_opt = optim.Adam(self.gen_net.parameters(), **gen_opt_params)
        self._dis_opt = optim.Adam(self.dis_net.parameters(), **dis_opt_params)

        # Define update criteria
        self._gan_crit = GANLoss("vanilla").to(self._device)
        self._mae_crit = nn.L1Loss()
        self._gen_lambda_mae = gen_lambda_mae

        # Logging
        self._pre_epoch = 0
        self._gan_epoch = 0
        self._pre_loss_meters = create_loss_meters(pretraining=True)
        self._gan_loss_meters = create_loss_meters()

    # === Training ===

    def train(self, train_dl: LabImageDataLoader, val_dl: LabImageDataLoader,
              n_epochs=20, display_every=5, mode="gan", checkpoints=None):

        # Choose optimization strategy
        assert mode in ("pre", "gan")
        optimize = getattr(self, f"_{mode}_optimize")
        loss_meters = getattr(self, f"_{mode}_loss_meters")
        last_epoch = getattr(self, f"_{mode}_epoch")

        # Set checkpointing arguments
        checkpoint, cp_after_each, cp_overwrite = set_cp_args(checkpoints)

        # --- Main training loop ---
        for curr_epoch in range(n_epochs):
            # Skip previous epochs when loaded from checkpoint
            if last_epoch > curr_epoch:
                continue

            reset_loss_meters(loss_meters)
            self.run_epoch(optimize, loss_meters, train_dl)

            # Make checkpoint
            last_epoch = curr_epoch + 1  # note that it's not linked to `self` since it's a primitive data type
            setattr(self, f"_{mode}_epoch", last_epoch)
            if checkpoint and last_epoch % cp_after_each == 0:
                cp_save_as = checkpoint + f"_epoch_{last_epoch:02d}"
                self.save_model(cp_save_as, cp_overwrite)

            # Give an update to performance
            if last_epoch % display_every == 0:
                # Print status
                print(f"Epoch {last_epoch}/{n_epochs}")
                log_results(loss_meters)
                # Visualize generated images
                self.evaluate(val_dl, mode, last_epoch)

        return self
    def run_epoch(self, optimize, loss_meters, train_dl):
        for i, batch in tqdm(enumerate(train_dl)):
            # Get real and predict (fake) image batch
            L = batch.L.to(self._device)
            ab = batch.ab.to(self._device)
            real_imgs = torch.cat([L, ab], dim=1)
            fake_imgs = torch.cat([L, self.gen_net(L)], dim=1)
            # enforce zero loss at padded values
            fake_imgs.masked_fill_(batch.pad_mask.to(self._device), batch.pad_fill_value)

            # Optimize
            loss_dict = optimize(real_imgs, fake_imgs)
            update_loss_meters(loss_meters, loss_dict, len(batch))

    def evaluate(self, val_dl, mode, last_epoch):
        self.gen_net.eval()
        val_batch = next(iter(val_dl))
        val_pred_ab = self.gen_net(val_batch.L.to(self._device)).to("cpu")
        pred_imgs = LabImageBatch(L=val_batch.L, ab=val_pred_ab, pad_mask=val_batch.pad_mask)
        pred_imgs.visualize(other=val_batch, show=False, save=True,
                            fname=f"{mode}_epoch_{last_epoch}_{time.time()}.png")
        self.gen_net.train()


    def _pre_optimize(self, real_imgs: torch.Tensor, fake_imgs: torch.Tensor) -> dict:
        self.gen_net.train()
        set_requires_grad(self.dis_net, False)
        loss = self._mae_crit(real_imgs[:, 1:], fake_imgs[:, 1:])  # use `ab` part only
        self._gen_opt.zero_grad()
        loss.backward()
        self._gen_opt.step()

        # Return loss
        loss_dict = {"mae_loss": loss}
        return loss_dict

    def _gan_optimize(self, real_imgs: torch.Tensor, fake_imgs: torch.Tensor) -> dict:
        # Update discriminator
        self.dis_net.train()
        set_requires_grad(self.dis_net, True)
        dis_loss, dis_loss_dict = self._dis_loss(real_imgs, fake_imgs)
        self._dis_opt.zero_grad()
        dis_loss.backward()
        self._dis_opt.step()

        # Update generator
        self.gen_net.train()
        set_requires_grad(self.dis_net, False)
        gen_loss, gen_loss_dict = self._gen_loss(real_imgs, fake_imgs)
        self._gen_opt.zero_grad()
        gen_loss.backward()
        self._gen_opt.step()

        # Return losses
        loss_dict = {**dis_loss_dict, **gen_loss_dict}
        return loss_dict

    def _dis_loss(self, real_imgs: torch.Tensor, fake_imgs: torch.Tensor) \
            -> tuple[torch.Tensor, LossMeterDict]:
        real_preds = self.dis_net(real_imgs)
        fake_preds = self.dis_net(fake_imgs.detach())
        real_loss = self._gan_crit(real_preds, True)
        fake_loss = self._gan_crit(fake_preds, False)
        dis_loss = (real_loss + fake_loss) / 2.
        # Logging
        loss_dict = dict(dis_loss_real=real_loss, dis_loss_fake=fake_loss, dis_loss=dis_loss)
        return dis_loss, loss_dict

    def _gen_loss(self, real_imgs: torch.Tensor, fake_imgs: torch.Tensor) \
            -> tuple[torch.Tensor, LossMeterDict]:
        fake_preds = self.dis_net(fake_imgs)
        gan_loss = self._gan_crit(fake_preds, False)
        mae_loss = self._mae_crit(real_imgs[:, 1:], fake_imgs[:, 1:])  # use `ab` part only
        gen_loss = gan_loss + mae_loss * self._gen_lambda_mae
        # Logging
        loss_dict = dict(gen_loss_gan=gan_loss, gen_loss_mae=mae_loss, gen_loss=gen_loss)
        return gen_loss, loss_dict

    def visualize_example_batch(self, real_imgs):
        pred_imgs = LabImageBatch(L=real_imgs.L, ab=self(real_imgs.L), pad_mask=real_imgs.pad_mask)
        pred_imgs.visualize(other=real_imgs, show=False, save=True)

    # === Evaluation ===

    def __call__(self, L: torch.Tensor):
        ab = self.gen_net(L.to(self._device)).to("cpu")
        return ab

    # === Save and load model ===

    @property
    def _save_dict(self):
        return {
            "torch": dict(  # Model networks and optimizers
                gen_model_state_dict="gen_net",
                dis_model_state_dict="dis_net",
                pre_optim_state_dict="_pre_opt",
                gen_optim_state_dict="_gen_opt",
                dis_optim_state_dict="_dis_opt",
            ),
            "other": dict(  # Logging
                pre_epoch="_pre_epoch",
                gan_epoch="_gan_epoch",
                pre_loss_meters="_pre_loss_meters",
                gan_loss_meters="_gan_loss_meters",
            )}

    def save_model(self, save_as: str, overwrite=True):
        save_as = secure_cp_path(save_as)
        if Path(save_as).exists() and not overwrite:
            warnings.warn(f"Model {save_as} not saved. Overwriting prohibited.")
        else:
            save_dict = self._save_dict
            save_dict["torch"] = {name: getattr(self, attr_name).state_dict()
                                  for name, attr_name in save_dict["torch"].items()}
            save_dict["other"] = {name: getattr(self, attr_name)
                                  for name, attr_name in save_dict["other"].items()}
            torch.save(save_dict, save_as)

    def load_model(self, load_from: str):
        save_dict = self._save_dict
        checkpoint = torch.load(load_from)
        for name, attr_name in save_dict["torch"].items():
            getattr(self, attr_name).load_state_dict(checkpoint["torch"][name])
        for name, attr_name in save_dict["other"].items():
            setattr(self, attr_name, checkpoint["other"][name])
        return self

class ImageGANAgentwFeedback(ImageGANAgent):
    def __init__(self, gen_net = None, *args, **kwargs):
        if gen_net is None:
            gen_net = build_res_u_net(3, 2)
        super().__init__(gen_net = gen_net, *args, **kwargs)

    def train(self, train_dl: LabImageDataLoader, val_dl: LabImageDataLoader,
              n_epochs=20, display_every=5, mode="gan", checkpoints=("checkpoints/gan", 10, True), sizes = [64,128]):

        # Choose optimization strategy
        assert mode in ("pre", "gan")
        optimize = getattr(self, f"_{mode}_optimize")
        loss_meters = getattr(self, f"_{mode}_loss_meters")
        last_epoch = getattr(self, f"_{mode}_epoch")

        # Set checkpointing arguments
        checkpoint, cp_after_each, cp_overwrite = set_cp_args(checkpoints)

        # --- Main training loop ---
        for curr_epoch in range(n_epochs):

            reset_loss_meters(loss_meters)
            self.run_epoch(optimize, loss_meters, train_dl, sizes)

            # Make checkpoint
            last_epoch = curr_epoch + 1  # note that it's not linked to `self` since it's a primitive data type
            setattr(self, f"_{mode}_epoch", last_epoch)
            if checkpoint and last_epoch % cp_after_each == 0:
                cp_save_as = checkpoint + f"_epoch_{last_epoch:02d}"
                self.save_model(cp_save_as, cp_overwrite)

            
            # Print status
            print(f"Epoch {last_epoch}/{n_epochs}")
            log_results(loss_meters)
        # Visualize generated images
        self.evaluate(val_dl, mode, last_epoch, sizes)

        return self

    def run_epoch(self, optimize, loss_meters, train_dl, sizes):
        for i, batch in tqdm(enumerate(train_dl)):

            prev_pred_imgs = torch.zeros([batch.L.shape[0], 2, sizes[0], sizes[0]]).to(self._device)
            for size in sizes:
                prev_pred_imgs, real_imgs, fake_imgs = self.one_iteration(batch, size, prev_pred_imgs)
                loss_dict = optimize(real_imgs, fake_imgs)
                update_loss_meters(loss_meters, loss_dict, len(batch))

    def one_iteration(self, batch, size, prev_pred_imgs):
        transforms = [
            # Uncomment for significant speed up
            T.Resize((size, size), T.InterpolationMode.BICUBIC),  # ATTENTION: This skews/distorts the images!
        ]

        transforms = T.Compose([*transforms])

        L = transforms(batch.L).to(self._device)
        ab = transforms(batch.ab).to(self._device)
        iter_input = torch.cat([L, transforms(prev_pred_imgs)], dim=1)

        prev_pred_imgs = self.gen_net(iter_input)
        real_imgs = torch.cat([L, ab], dim=1)
        fake_imgs = torch.cat([L, prev_pred_imgs], dim=1)

        fake_imgs.masked_fill_(transforms(batch.pad_mask).to(self._device), batch.pad_fill_value)
        return prev_pred_imgs.detach(), real_imgs, fake_imgs

    def evaluate(self, val_dl, mode, last_epoch, sizes):
        self.gen_net.eval()
        val_batch = next(iter(val_dl))
        pred_imgs = self.colorize_images(val_batch, sizes)
        pred_imgs.visualize(other=val_batch, show=False, save=True,
                            fname=f"{mode}_epoch_{last_epoch}_{time.time()}.png")
        self.gen_net.train()


    def colorize_images(self, batch: LabImageBatch, sizes: list[int] = [64, 128]):
        self.gen_net.eval()
        L = batch.L
        prev_pred_imgs = torch.zeros([L.shape[0], 2, sizes[0], sizes[0]]).to(self._device)
        for size in sizes:
            transforms = [
                # Uncomment for significant speed up
                T.Resize((size, size), T.InterpolationMode.BICUBIC),  # ATTENTION: This skews/distorts the images!
            ]

            transforms = T.Compose([*transforms])
            iter_input = torch.cat([transforms(L).to(self._device), transforms(prev_pred_imgs)], dim=1)

            prev_pred_imgs = self.gen_net(iter_input)

        return LabImageBatch(L=transforms(L).to("cpu"), ab=prev_pred_imgs.to("cpu"), pad_mask=transforms(batch.pad_mask))

    def colorize_image(self, img: LabImage, sizes: list[int] = [64, 128]):
        L =torch.unsqueeze(img.L, 0)
        self.gen_net.eval()
        prev_pred_imgs = torch.zeros([L.shape[0], 2, sizes[0], sizes[0]]).to(self._device)
        for size in sizes:
            transforms = [
                # Uncomment for significant speed up
                T.Resize((size, size), T.InterpolationMode.BICUBIC),  # ATTENTION: This skews/distorts the images!
            ]

            transforms = T.Compose([*transforms])
            iter_input = torch.cat([transforms(L).to(self._device), transforms(prev_pred_imgs)], dim=1)

            prev_pred_imgs = self.gen_net(iter_input)
        final_prediction = prev_pred_imgs.to("cpu")
        return LabImage(L=transforms(L).to("cpu")[0], ab=final_prediction[0])

    def visualize_example_batch(self, real_imgs, sizes = [64,128]):
        pred_imgs = self.colorize_images(real_imgs, sizes)
        pred_imgs.visualize(other=real_imgs, show=False, save=True)