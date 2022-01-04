"""This is the main function of our project.

Hint:
    The ``--help`` option depends on ``--model`` and, hence,
    only shows all arguments when ``--model`` is provided.
"""

import warnings
from typing import Type
import numpy as np
import torch
from pytorch_lightning import Trainer
from src.gan import BaseModule, PreTrainer, ImageGAN, C2FImageGAN
from src.utils.callbacks import get_callbacks_for_given_model
from src.utils.data import URLs, ColorizationDataModule

from argparse import ArgumentParser, RawDescriptionHelpFormatter

# Fix random states!
np.random.seed(2109385902)
torch.manual_seed(923845902387)


def model_cls(model: str) -> Type[BaseModule]:
    if model == "pretrain":
        return PreTrainer
    elif model == "base":
        return ImageGAN
    elif model == "c2f":
        return C2FImageGAN
    else:
        warnings.warn(f"Model ``{model}`` not recognized or empty.")
        return BaseModule  # return dummy module


def make_parser(hard_args=None) -> ArgumentParser:

    # --- Add PROGRAM level args ---
    parent_parser = ArgumentParser(add_help=False)  # suppress help, so we print all options in response to ``--help``
    parent_parser.add_argument("--model", type=str, help="``pretrain``, ``base`` or ``c2f``")
    parent_parser.add_argument("--pretrained_ckpt_path", type=str,
                               help="path to model checkpoint; restore just the weights")
    parent_parser.add_argument("--full_ckpt_path", type=str,
                               help="path to model checkpoint; restore the full training ")

    # --- Add MODEL args ---
    parent_parser = ColorizationDataModule.add_argparse_args(parent_parser)
    # add model type specific args
    inter_args, _ = parent_parser.parse_known_args(hard_args)
    parent_parser = model_cls(inter_args.model).add_model_specific_args(parent_parser)

    # --- Add TRAINER args ---
    parser = ArgumentParser(parents=[parent_parser],
                            description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser = Trainer.add_argparse_args(parser)

    # --- Set default args ---
    parser.set_defaults(
        data_dir=URLs.COCO_SAMPLE,
        seed=123456789,
    )

    return parser


def adjust_args(args):

    # Add callbacks
    args.callbacks = get_callbacks_for_given_model(args)

    # Modify batch_size according to how pytorch-lightning scales it
    if args.gpus is not None:
        if args.gpus == -1:
            num_gpus = torch.cuda.device_count()
        elif isinstance(args.gpus, str):
            num_gpus = len(args.gpus.split(","))
        elif isinstance(args.gpus, (tuple, list)):
            num_gpus = len(args.gpus)
        else:
            raise NotImplementedError
        args.batch_size = int(args.batch_size / num_gpus)

    return args


def main(args):
    args = adjust_args(args)

    # Setup
    dm = ColorizationDataModule.from_argparse_args(args)
    module = model_cls(args.model)

    # Training
    model = module(**vars(args))
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=dm, ckpt_path=args.full_ckpt_path)

    # Test (only right before publishing your paper or pushing to production!)
    # trainer.test(ckpt_path="best", datamodule=dm)

    print_args = [f"{a}={getattr(args, a)}" for a in vars(args)]
    print(f"\nPassed arguments: {', '.join(print_args)}")


if __name__ == '__main__':
    # TODO: Remove ``hard_args`` before project submission
    # hard_args = ("--model pretrain "
    #              # "--pretrained_ckpt_path lightning_logs/version_12/checkpoints/epoch=4-step=29.ckpt "
    #              "--batch_size 4 --dataset_size 32 --max_img_size 64 "
    #              # "--gen_net_params 3 2 128 "
    #              "--max_epochs 5 --log_every_n_steps 5 --check_val_every_n_epoch 5 "
    #              ).split()
    hard_args = None
    p_args = make_parser(hard_args).parse_args(hard_args)
    main(p_args)
