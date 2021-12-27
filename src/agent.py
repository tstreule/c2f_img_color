import warnings
from pathlib import Path
from tqdm import tqdm
import time

import torch
from torch import nn, optim
from kornia.losses import ssim_loss, psnr_loss

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
                      "gen_loss_gan", "gen_loss_mae", "gen_loss",
                      "ssim_loss", "psnr_loss"]
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
        print(f"{loss_name: <{fill}}: {loss_meter.mean:.4f} ± {loss_meter.std:.4f}")


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

            self._run_epoch(optimize, loss_meters, train_dl)
            reset_loss_meters(loss_meters)

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
                val_batch = next(iter(val_dl))
                self.visualize_example_batch(val_batch, show=False, save=True,
                                             fname=f"{mode}_epoch_{last_epoch}_{time.time()}.png")

        return self

    def _run_epoch(self, optimize, loss_meters, train_dl):
        for i, batch in tqdm(enumerate(train_dl)):
            # Get real and predict (fake) image batch
            real_imgs = batch.lab.to(self._device)
            fake_imgs = self(real_imgs[:, :1])  # equivalent to batch.L but faster
            # enforce zero loss at padded values
            fake_imgs.masked_fill_(batch.pad_mask.to(self._device), batch.pad_fill_value)

            # Optimize
            loss_dict = optimize(real_imgs, fake_imgs)
            update_loss_meters(loss_meters, loss_dict, len(batch))

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
        loss_dict = {"dis_loss_real": real_loss, "dis_loss_fake": fake_loss, "dis_loss": dis_loss}
        return dis_loss, loss_dict

    def _gen_loss(self, real_imgs: torch.Tensor, fake_imgs: torch.Tensor) \
            -> tuple[torch.Tensor, LossMeterDict]:
        fake_preds = self.dis_net(fake_imgs)
        gan_loss = self._gan_crit(fake_preds, False)
        mae_loss = self._mae_crit(real_imgs[:, 1:], fake_imgs[:, 1:])  # use `ab` part only
        gen_loss = gan_loss + mae_loss * self._gen_lambda_mae
        # Logging
        ssim_loss_ = ssim_loss(real_imgs, fake_imgs, 5)  # is symmetric
        psnr_loss_ = psnr_loss(real_imgs, fake_imgs, 1.)  # is not(?) symmetric
        loss_dict = {"gen_loss_gan": gan_loss, "gen_loss_mae": mae_loss, "gen_loss": gen_loss,
                     "ssim_loss": ssim_loss_, "psnr_loss": psnr_loss_}
        return gen_loss, loss_dict

    # === Generate ===

    def __call__(self, L: torch.Tensor) -> torch.Tensor:
        L = L.to(self._device)
        pred_imgs = torch.cat([L, self.gen_net(L)], dim=1)
        return pred_imgs

    # === Evaluation ===

    def visualize_example_batch(self, real_imgs: LabImageBatch, **kwargs):
        prev_train_mode = self.gen_net.training
        self.gen_net.eval()
        pred_lab = self(real_imgs.L).to("cpu")
        pred_imgs = LabImageBatch(lab=pred_lab, pad_mask=real_imgs.pad_mask)
        pred_imgs.visualize(other=real_imgs, **kwargs)
        self.gen_net.train(prev_train_mode)

    @property
    def loss_meters(self) -> dict[str, LossMeterDict]:
        return {"pre": self._pre_loss_meters, "gan": self._gan_loss_meters}

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


class C2FImageGANAgent(ImageGANAgent):

    def __init__(self, *args, gen_net_params=(3, 2, 128), shrink_size=2.0,
                 min_ax_size=64, max_c2f_depth=5, **kwargs):
        super().__init__(*args, gen_net_params=gen_net_params,  **kwargs)
        # Define shrink size parameter (larger values speed up training and prediction)
        self.shrink_size = shrink_size
        # Define minimal size an image axis must have for recursion
        self.min_ax_size = min_ax_size
        # Define maximal recursive depth
        self.max_c2f_depth = max_c2f_depth

    # === Training ===

    def _run_epoch(self, optimize, loss_meters, train_dl):
        for i, batch in tqdm(enumerate(train_dl)):
            # Get real images
            real_imgs = batch.lab.to(self._device)
            # Optimization is done inside recursive loop
            pred_imgs = self._c2f_recursive(real_imgs, opt=(optimize, loss_meters, batch))

    def _c2f_recursive(self, real_imgs: torch.Tensor, opt=None, rec_depth=0) -> torch.Tensor:
        real_sizes = real_imgs.shape[2:]
        smaller_sizes = [int(size / self.shrink_size) for size in real_sizes]

        if any(size < self.min_ax_size for size in smaller_sizes) \
                or rec_depth >= self.max_c2f_depth:
            # Initialize dummy predictions
            # when not training `real_imgs` can also be just a "L"
            prev_pred_imgs = torch.zeros(real_imgs.shape[0], 3, *real_imgs.shape[2:])
        else:
            resize = T.Resize(tuple(smaller_sizes))
            prev_pred_imgs = self._c2f_recursive(resize(real_imgs), opt, rec_depth+1)

        # Resizer for scaling up or down
        resize = T.Resize(tuple(real_sizes))

        # Prediction
        L = real_imgs[:, :1].to(self._device)
        ab = resize(prev_pred_imgs.detach()[:, 1:]).to(self._device)
        pred_input = torch.cat([L, ab], dim=1)
        pred_ab = self.gen_net(pred_input)
        pred_imgs = torch.cat([L, pred_ab], dim=1)

        # Optimize
        if opt is not None:
            optimize, loss_meters, batch = opt
            # Enforce zero loss at padded values
            pred_imgs.masked_fill_(resize(batch.pad_mask).to(self._device), batch.pad_fill_value)
            # Optimize
            loss_dict = optimize(real_imgs, pred_imgs)
            if rec_depth == 0:  # don't log recursion losses
                update_loss_meters(loss_meters, loss_dict, len(batch))

        return pred_imgs

    # === Generate ===

    def __call__(self, L: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            L = L.to(self._device)
            pred_imgs = self._c2f_recursive(L).to("cpu")
        return pred_imgs

    def colorize_image_batch(self, lab_img: LabImageBatch):
        pred = self(lab_img.L)
        return LabImageBatch(lab= pred)

    def colorize_image(self, lab_img: LabImage):
        batch = LabImageBatch([lab_img])
        pred = self(batch.L)
        return LabImage(lab= pred[0])

