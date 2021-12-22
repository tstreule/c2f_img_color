# This is the main function of our project

import numpy as np
import torch

from src.agent import ImageGANAgent,ImageGANAgentwFeedback
from src.utils.data import URLs, get_image_paths, make_dataloader
from src.utils.image import *

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
# >>> Pretraining
parser.add_argument("--pre-cp-name", type=str, default=None,
                    help="relative path to U-Net checkpoint; will be joined with `--cp-dir`")
parser.add_argument("--pre-num-epochs", type=int, default=20,
                    help="number of training pretraining epochs for U-Net")
# >>> GAN
parser.add_argument("--gan-cp-name", type=str, default=None,
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
    hard_args = "--dataset-size 32 --pre-num-epochs 1 --gan-num-epochs 1 --batch-size 2 "
    hard_args += "--cp-dir checkpoints/base/ "
    #hard_args += "--pre-cp pre_final.pt "
    #hard_args += "--gan-cp gan_final.pt "
    args = parser.parse_args(hard_args.split())

    print_args = [f"{a}={getattr(args, a)}" for a in vars(args)]
    print(f"Passed arguments: {', '.join(print_args)}")

    # ---------------------------
    # Create dataset and dataloaders

    print("\n=====================")
    print("Getting dataset...")
    train_paths, test_paths = get_image_paths(args.dataset, args.dataset_size, test=args.test_split)
    train_dl = make_dataloader(args.batch_size, args.num_workers,
                               paths=train_paths, split="train", rng=torch_rng)
    val_dl = make_dataloader(args.batch_size, args.num_workers,
                             paths=test_paths, split="test")
    print(" ...done")

    # ---------------------------
    # Training

    print("\n=====================")
    print("Initialize agent...")
    agent = ImageGANAgent()
    print(" ...done")

    # Set default checkpoint args
    pre_cps = (args.cp_dir + "pre", 10, args.cp_overwrite)
    gan_cps = (args.cp_dir + "gan", 10, args.cp_overwrite)
    pre_cp_final = args.cp_dir + "pre_final", args.cp_overwrite
    gan_cp_final = args.cp_dir + "gan_final", args.cp_overwrite

    # Pretraining of U-Net
    if args.gan_cp_name is None:
        print("\n=====================")
        if args.pre_cp_name is not None:
            print("Loading pretrained U-Net from checkpoint...")
            agent.load_model(args.cp_dir + args.pre_cp_name)
        print("Pretraining U-Net...")
        agent.train(train_dl, val_dl, args.pre_num_epochs, mode="pre", checkpoints=pre_cps)
        agent.save_model(*pre_cp_final)
        print(" ...done")

    # Training of GAN
    print("\n=====================")
    if args.gan_cp_name is not None:
        print("Loading GAN from checkpoint...")
        agent.load_model(args.cp_dir + args.gan_cp_name)
    print("Training GAN...")
    agent.train(train_dl, val_dl, args.gan_num_epochs, mode="gan", checkpoints=gan_cps)
    agent.save_model(*gan_cp_final)
    print(" ...done")

    # ---------------------------
    # Evaluation

    print("\n=====================")
    print("Evaluation...")

    # # Retrieve trained generator
    # # -> you can directly call `agent(batch.L)`
    # generator = agent.gen_net
    # generator.eval()

    # Visualize example batch
    real_imgs = next(iter(val_dl))
    agent.visualize_example_batch(real_imgs)
    # pred_imgs[0].visualize(real_imgs[0], show=False, save=True)

    print(" ...done")


if __name__ == "__main__":
    main()
