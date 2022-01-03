# Iterative Coarse-to-Fine Image Colorization with GAN

Semester Project for the class Deep Learning in autumn semester 2021.


## Setup

We recommend to start with a conda environment and install its requirements as follows

```bash
# Create conda env
conda create -n dl-env python=3.9
# Activate conda env
source activate dl-env
# Install dependencies
conda install --file requirements.txt
```

### TensorBoard

Logging with TensorBoard comes on the hood. While or after training you can open TensorBoard in the browser after the command `tensorboard --logdir ./lightning_logs`.
