import torch

from fastai.vision.learner import create_body
from torchvision.models.resnet import resnet18
from fastai.vision.models.unet import DynamicUnet


__all__ = ["build_res_u_net"]


def build_res_u_net(n_input=1, n_output=2, size=256, arch="resnet18", pretrained=True):
    """First pretraining step for generator -> Use a pretrained U-Net

    Args:
        n_input: Image input dimension
        n_output: Image output dimension
        size:
        arch:
        pretrained:

    Returns:
        (Pretrained) U-Net
    """
    arch_dict = dict(
        resnet18=resnet18,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body = create_body(arch_dict[arch], pretrained=pretrained, n_in=n_input, cut=-2)
    gen_net = DynamicUnet(body, n_output, (size, size)).to(device)
    return gen_net
