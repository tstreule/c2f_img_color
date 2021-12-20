# This is the main function of our project

import numpy as np
import torch

from src.gan import ImageGAN
from src.generator import *
from src.utils.image import *
from src.utils.dataset import *


# Fix random states!
np.random.seed(2109385902)
torch.manual_seed(923845902387)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Create dataset and dataloaders
    print("Getting dataset")
    train_paths, test_paths = get_image_paths(URLs.COCO_SAMPLE, 900, test=0.2)
    train_dl = make_dataloader(8, paths=train_paths, split="train")
    val_dl = make_dataloader(8, paths=test_paths, split="test")
    print("Done")
    # ---------------------------
    # Training

    # Pre-train generator
    print("Pretraining generator")
    generator = build_res_u_net(n_input=1, n_output=2, size=128)
    pretrain_generator(generator, train_dl, epochs=10)
    print("Done")

    # Train with GAN training agent
    agent = ImageGAN(gen_net=generator)
    agent.train(train_dl, epochs=20)

    # ---------------------------
    # Evaluation

    # Retrieve trained generator
    generator = agent.gen_net

    # Visualize example batch
    real_imgs = next(iter(val_dl))
    pred_imgs = LabImageBatch(L=real_imgs.L, ab=generator(real_imgs.L.to(device)).to("cpu"))
    pred_imgs.visualize(other=real_imgs, save="True")

    # # ---------------------------
    # # Playground
    #
    # # Print batch data
    # batch = next(iter(train_dl))
    # Ls, abs_ = batch.L, batch.ab
    # print(Ls.shape, abs_.shape)
    # print(len(train_dl), len(val_dl))
    #
    # # Visualize
    # batch[0].visualize()
    # batch.visualize(draw_n=5)

    agent.save_model()
if __name__ == "__main__":
    main()
