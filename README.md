# Iterative Coarse-to-Fine Image Colorization with GAN

A semester project for the class Deep Learning at ETH Zürich in autumn semester 2021.

In this paper, we propose a novel design to solve the problem of coloring high-resolution images whose colors should be consistent throughout the image.
Our main contribution is the idea of scaling and coloring input images "from coarse to fine" in an iterative manner. Instead of directly coloring a grayscale image, we first colorize a pixelated version and scale it up. The result of each iteration then serves as a color bias for the next larger pixelated version until we reach the final image size.

> ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+)
> **TODO:** Make a reference to our paper as soon as it's uploaded.

> ![#f03c15](https://via.placeholder.com/15/f03c15/000000?text=+)
> **TODO:** Include image here (e.g., model design <i>and/or</i> sample images).


## Prerequisites

- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN


## Getting Started

### Installation

- Clone this repo:
  ```bash
  git clone https://github.com/tstreule/c2f_img_color
  cd c2f_img_color
  ```

- Install [PyTorch](https://pytorch.org) and other dependencies:
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

### Training/testing

- Train a model:
  ```bash
  # Train base model
  python main.py --model base
  # Train C2F model
  python main.py --model c2f --gen_net_params 3 2 128
  ```

- Note that logging and model checkpoints per default will be saved in the folder `lightning_logs`.

- Test the model:<br>
  During or after the training you can keep track of the model performance via TensorBoard. Just start a localhost server with the command `tensorboard --logdir ./lightning_logs` and open the link in the browser.

### Apply a pre-trained model

- Please refer to the [project demo file](demo.ipynb) for instructions on how to apply a pretrained model to custom images.
- You can download our pretrained models from this [Polybox link](https://polybox.ethz.ch/index.php/s/uwF5Gml6rJjb0QY).


> **Tip:** Use `python main.py --help` to find parameters that can be tweaked.
> 
> Note that model-specific arguments are only visible when the model is specified (e.g., `python main.py --model base --help`).



## Authors

- [Timo Streule](https://github.com/tstreule), tstreule@ethz.ch
- [Thomas Rüegg](https://github.com/Thomacdebabo), rueeggth@ethz.ch
- [Marc Styger](https://github.com/stygerma), stygerma@ethz.ch


## Acknowledgments

Our code inspired by Zhu et al. [[2017]](https://arxiv.org/pdf/1703.10593.pdf) and Isola et al. [[2017]](https://arxiv.org/pdf/1611.07004.pdf).
