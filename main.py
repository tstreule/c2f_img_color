# This is the main function of our project

import numpy as np
import torch

from src.gan import ImageGAN
from src.generator import *
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------
    # Argument Parser

    # Uncomment when you want hard-coded parse args
    hard_args = "--dataset-size 32 --unet-num-epochs 1 --gan-num-epochs 1 "
    hard_args += "--cp-dir checkpoints/base/ "
   # hard_args += "--unet-cp unet_final.pt "
    #hard_args += "--gan-cp gan_final.pt "
    args = parser.parse_args(hard_args.split())

    print_args = [f"{a}={getattr(args, a)}" for a in vars(args)]
    print("Passed arguments:", ", ".join(print_args), "\n")

    # ---------------------------
    # Create dataset and dataloaders

    print("Getting dataset...")
    train_paths, test_paths = get_image_paths(args.dataset, args.dataset_size, test=args.test_split)
    train_dl = make_dataloader(args.batch_size, args.num_workers,
                               paths=train_paths, split="train", rng=torch_rng)
    val_dl = make_dataloader(args.batch_size, args.num_workers,
                             paths=test_paths, split="test")
    print(" ...done\n")

    # ---------------------------
    # Training

    if args.gan_cp is None:
        # Pretrain generator
        generator = build_res_u_net(n_input=1, n_output=2, size=args.unet_size)
        if args.unet_cp is None:
            print("Pretraining generator...")
            unet_cps = (args.cp_dir + "unet", 10, args.cp_overwrite)
            pretrain_generator(generator, train_dl, epochs=args.unet_num_epochs, checkpoints=unet_cps)
            unet_save = secure_cp_path(args.cp_dir + "unet_final")
            save_model(generator, unet_save, overwrite=args.cp_overwrite)
        else:
            print("Loading generator from checkpoint...")
            load_model(generator, args.cp_dir + args.unet_cp)
        print(" ...done\n")

        # Train with GAN training agent
        print("Training GAN...")
        agent = ImageGAN(gen_net=generator)
        gan_cps = (args.cp_dir + "gan_final", 10, args.cp_overwrite)
        agent.train(train_dl, epochs=args.gan_num_epochs, checkpoints=gan_cps)
        gan_save = secure_cp_path(args.cp_dir + "gan_final")
        agent.save_model(gan_save, overwrite=args.cp_overwrite)

    else:
        # Load GAN from checkpoint
        print("Loading GAN from checkpoint...")
        agent = ImageGAN()
        agent.load_model(args.cp_dir + args.gan_cp)

    print(" ...done\n")

    # ---------------------------
    # Evaluation

    # Retrieve trained generator
    generator = agent.gen_net
    generator.eval()

    # Visualize example batch
    real_imgs = next(iter(val_dl))
    pred_imgs = LabImageBatch(L=real_imgs.L, ab=generator(real_imgs.L.to(device)).to("cpu"))
    pred_imgs.padding = real_imgs.padding
    pred_imgs.visualize(other=real_imgs, save=True)
    pred_imgs[0].visualize(real_imgs[0], save=True)


if __name__ == "__main__":
    main()
