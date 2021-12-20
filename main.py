# This is the main function of our project

import numpy as np
import torch

from src.gan import ImageGAN
from src.generator import *
from src.utils.image import *
from src.utils.dataset import *

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
# Training
parser.add_argument("--num-unet-epochs", type=int, default=20,
                    help="number of training epochs for U-Net")
parser.add_argument("--num-gan-epochs", type=int, default=20,
                    help="number of training epochs for ImageGAN")
# Checkpoints: for checkpoints use either single or triple argument
parser.add_argument("--unet-checkpoint", type=str, default="basic/unet", nargs="+",
                    help="default folder for saving U-Net checkpoints")
parser.add_argument("--gan-checkpoint", type=str, default="basic/gan", nargs="+",
                    help="default folder for saving ImageGAN checkpoints")
parser.add_argument("--pretrained", dest="from_pretrained", action="store_true",
                    help="use pretrained ImageGAN (from `--gan-checkpoint`)")
parser.add_argument("--not-pretrained", dest="from_pretrained", action="store_false",
                    help="do not use a pretrained model")
parser.set_defaults(from_pretrained=False)
args = parser.parse_args()


# Fix random states!
np.random.seed(2109385902)
torch.manual_seed(923845902387)
torch_rng = torch.Generator()
torch_rng.manual_seed(209384575)


def main():
    # Create dataset and dataloaders
    train_paths, test_paths = get_image_paths(args.dataset, args.dataset_size, test=args.test_split)
    train_dl = make_dataloader(args.batch_size, args.num_workers,
                               paths=train_paths, split="train", rng=torch_rng)
    val_dl = make_dataloader(args.batch_size, args.num_workers,
                             paths=test_paths, split="test")

    # ---------------------------
    # Training

    if not args.from_pretrained:
        # Pre-train generator
        generator = build_res_u_net(n_input=1, n_output=2, size=256)
        pretrain_generator(generator, train_dl, epochs=args.num_unet_epochs,
                           load_from_checkpoint=False, checkpoint=args.unet_checkpoint)

        # Train with GAN training agent
        agent = ImageGAN(gen_net=generator)
        agent.train(train_dl, epochs=args.num_gan_epochs, checkpoint=args.gan_checkpoint)

    else:
        # Load from checkpoint
        agent = ImageGAN()
        agent.load_model(args.gan_checkpoint+"_epoch_01_final.pt")

    # ---------------------------
    # Evaluation

    # Retrieve trained generator
    generator = agent.gen_net
    generator.eval()

    # Visualize example batch
    real_imgs = next(iter(val_dl))
    pred_imgs = LabImageBatch(L=real_imgs.L, ab=generator(real_imgs.L))
    pred_imgs.padding = real_imgs.padding
    pred_imgs.visualize(other=real_imgs)


if __name__ == "__main__":
    # Uncomment when you want hard-coded parse args
    # args = "--dataset-size 32 --num-unet-epochs 1 --num-gan-epochs 1 --pretrained"
    # args = parser.parse_args(args.split())
    print("Passed arguments:")
    print(", ".join([f"{arg}={getattr(args, arg)}" for arg in vars(args)]), "\n")

    main()
