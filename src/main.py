# This is the main function of our project

import numpy as np
import torch

from utils.dataset import URLs, get_image_paths, make_dataloader


# Fix random states!
np.random.seed(2109385902)
torch.manual_seed(923845902387)


def main():
    # Create dataset and dataloaders
    train_paths, test_paths = get_image_paths(URLs.COCO_SAMPLE, 1_000, test=0.2)
    train_dl = make_dataloader(paths=train_paths, split="train")
    val_dl = make_dataloader(paths=test_paths, split="test")

    # ---------------------------
    # Playground

    # Print batch data
    batch = next(iter(train_dl))
    Ls, abs_ = batch.L, batch.ab
    print(Ls.shape, abs_.shape)
    print(len(train_dl), len(val_dl))

    # Visualize
    batch.visualize(draw_n=5)
    batch[0].visualize()

    return


if __name__ == "__main__":
    main()
