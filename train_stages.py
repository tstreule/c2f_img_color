import numpy as np
import torch

from src.agent import ImageGANAgent,ImageGANAgentwFeedback
from src.utils.data import URLs, get_image_paths, make_dataloader
from src.utils.image import *

# Fix random states!
np.random.seed(2109385902)
torch.manual_seed(923845902387)
torch_rng = torch.Generator()
torch_rng.manual_seed(209384575)


def main():
    print("\n=====================")
    print("Getting dataset...")
    train_paths, test_paths = get_image_paths(URLs.COCO_SAMPLE, 4096, test=0.2)
    train_dl = make_dataloader(16, 4,
                               paths=train_paths, split="train", rng=torch_rng)
    val_dl = make_dataloader(8, 4,
                             paths=test_paths, split="test")
    print(" ...done")
    checkpoint_path = r"checkpoints/gan_epoch_60.pt"
    print("\n=====================")
    print("Initialize agent...")
    agent = ImageGANAgentwFeedback()
    agent.load_model(checkpoint_path)
    print(" ...done")

    print("Training GAN...")
    agent.train(train_dl, val_dl, 40,sizes=[32,64,128], mode="pre")
    agent.evaluate(val_dl, "pre", 100, [128, 256])
    agent.train(train_dl, val_dl, 50,sizes=[32,64,128], mode="gan")
    agent.train(train_dl, val_dl, 50, sizes=[64,128], mode="gan")
    agent.evaluate(val_dl, "stage1", 100, [128, 256])
    train_dl = make_dataloader(4, 4,paths=train_paths, split="train", rng=torch_rng)
    val_dl = make_dataloader(2, 4, paths=test_paths, split="test")

    agent.train(train_dl, val_dl, 20,sizes=[64,128,256], mode="gan")
    train_dl = make_dataloader(2, 4,paths=train_paths, split="train", rng=torch_rng)
    val_dl = make_dataloader(2, 4, paths=test_paths, split="test")
    agent.train(train_dl, val_dl, 20,sizes=[128,256,512], mode="gan")
    agent.evaluate(val_dl,"gan", 100, [128,256])
    agent.save_model("checkpoints/train_stages.pt", overwrite=True)

    print(" ...done")

if __name__ == "__main__":
    main()
