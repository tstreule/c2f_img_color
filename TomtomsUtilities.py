from src.utils.dataset import *
import torch
import numpy as np
def get_dataloaders(torch_rng, args):
    print("Getting dataset...")
    train_paths, test_paths = get_image_paths(args.dataset, args.dataset_size, test=args.test_split)
    train_dl = make_dataloader(args.batch_size, args.num_workers,
                               paths=train_paths, split="train", rng=torch_rng)
    val_dl = make_dataloader(args.batch_size, args.num_workers,
                             paths=test_paths, split="test")
    print(" ...done\n")
    return train_dl, val_dl

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_sizes_from_img(img):
    max2 = int(np.round(np.log2(max(img.shape))))
    return np.power(2,list(range(max2))[-3:])