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
    # Create dataset and dataloaders
    train_paths, test_paths = get_image_paths(URLs.COCO_SAMPLE, 40, test=0.2)
    train_dl = make_dataloader(16, paths=train_paths, split="train")
    val_dl = make_dataloader(16, paths=test_paths, split="test")

    # ---------------------------
    # Training

    u_net_checkpoint = "basic/u_net"
    gan_checkpoint = "basic/gan"
    from_scratch = False

    if from_scratch:
        # Pre-train generator
        generator = build_res_u_net(n_input=1, n_output=2, size=256)
        pretrain_generator(generator, train_dl, epochs=1,
                           load_from_checkpoint=False, checkpoint=u_net_checkpoint)

        # Train with GAN training agent
        agent = ImageGAN(gen_net=generator)
        agent.train(train_dl, epochs=1, checkpoint=gan_checkpoint)

    else:
        agent = ImageGAN()
        agent.load_model(gan_checkpoint+"_epoch_01_final.pt")

    # ---------------------------
    # Evaluation

    # Retrieve trained generator
    generator = agent.gen_net
    generator.eval()

    # Visualize example batch
    real_imgs = next(iter(val_dl))
    pred_imgs = LabImageBatch(L=real_imgs.L, ab=generator(real_imgs.L))
    pred_imgs.visualize(other=real_imgs)

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


if __name__ == "__main__":
    main()
