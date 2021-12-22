# This is the main function of our project
import TomtomsUtilities as ttu

import numpy as np
import torch

from src.gan import ImageGAN, ImageGANwFeedback
from src.generator import *
from src.generator import pretrain_generator_w_feedback
from src.utils.image import *
from src.utils.dataset import *
from src.utils.checkpoint import load_model, save_model, secure_cp_path

import argparse

parser = argparse.ArgumentParser()
# Dataset
parser.add_argument("--dataset", type=str, default=URLs.COCO_SAMPLE,
                    help="name of a `fastai` dataset")
parser.add_argument("--dataset-size", type=int, default=10_000,
                    help="number of images to draw from dataset")
parser.add_argument("--test-split", type=float, default=0.2,
                    help="percentage of dataset to become train split")
# DataLoader
parser.add_argument("--batch-size", type=int, default=16,
                    help="batch size of data loaders")
parser.add_argument("--num-workers", type=int, default=4,
                    help="number of workers for data loaders")
# Checkpoints
parser.add_argument("--cp-dir", type=str, default="checkpoints/",
                    help="all model checkpoints land in this directory")
parser.add_argument("--cp-overwrite", dest="cp_overwrite", action="store_true",
                    help="overwrite model checkpoints if name already exists")
parser.add_argument("--cp-not-overwrite", dest="cp_overwrite", action="store_false")
parser.set_defaults(cp_overwrite=True)
# U-Net
parser.add_argument("--unet-cp", type=str, default=None,
                    help="relative path to U-Net checkpoint; will be joined with `--cp-dir`")
parser.add_argument("--unet-size", type=int, default=128,
                    help="size of U-Net")
parser.add_argument("--unet-num-epochs", type=int, default=20,
                    help="number of training epochs for U-Net")
# Image GAN
parser.add_argument("--gan-cp", type=str, default=None,
                    help="relative path to ImageGAN checkpoint; will be joined with `--cp-dir`")
parser.add_argument("--gan-num-epochs", type=int, default=20,
                    help="number of training epochs for ImageGAN")
args = parser.parse_args()


# Fix random states!
np.random.seed(2109385902)
torch.manual_seed(923845902387)
torch_rng = torch.Generator()
torch_rng.manual_seed(209384575)


def main():
    # ---------------------------
    # Argument Parser

    # Uncomment when you want hard-coded parse args
    hard_args = "--dataset-size 4096 --unet-num-epochs 0 --gan-num-epochs 20 --batch-size 8 "
    hard_args += "--cp-dir checkpoints/base/ "
   # hard_args += "--unet-cp unet_final.pt "
    hard_args += "--gan-cp gan_epoch_20.pt"
    args = parser.parse_args(hard_args.split())

    print_args = [f"{a}={getattr(args, a)}" for a in vars(args)]
    print("Passed arguments:", ", ".join(print_args), "\n")

    # ---------------------------
    # Create dataset and dataloaders
    train_dl, val_dl = ttu.get_dataloaders(torch_rng, args)
    agent = ImageGANwFeedback()

    if args.gan_cp is not None:
        print("Loading GAN from checkpoint...")
        agent.load_model(args.cp_dir + args.gan_cp)

    if args.unet_cp is not None:
        print("Loading generator from checkpoint...")
        agent.load_generator(args.cp_dir + args.unet_cp)

    if args.unet_num_epochs != 0:
        print("Pretraining generator...")
        unet_cps = (args.cp_dir + "unet", 10, args.cp_overwrite)
        unet_save = secure_cp_path(args.cp_dir + "unet_final")

        agent.pretrain_generator( train_dl, epochs=args.unet_num_epochs, checkpoints=unet_cps)
        agent.save_generator(unet_save, overwrite=args.cp_overwrite)

    if args.gan_num_epochs != 0:
        print("Training GAN...")
        gan_cps = (args.cp_dir + "gan_final", 10, args.cp_overwrite)
        agent.train(train_dl,val_dl, epochs=args.gan_num_epochs, checkpoints=gan_cps, sizes=[64,128,256])
        gan_save = secure_cp_path(args.cp_dir + "gan_final")
        agent.save_model(gan_save, overwrite=args.cp_overwrite)

    print(" ...done\n")

    print("Evaluating")
    agent.evaluate_model(val_dl)









if __name__ == "__main__":
    main()
