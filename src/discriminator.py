import torch
from torch import nn


__all__ = ["PatchDiscriminator"]


class PatchDiscriminator(nn.Module):

    def __init__(self, in_channels: int, num_filters=64, n_down=3):
        """Implements a model by stacking blocks of Conv-BatchNorm-LeakyReLU
        to decide whether the input image is fake or real.

        Note that, in comparison to a 'vanilla' discriminator, the model outputs
        one number for every patch (receptive field). This choice seems reasonable
        because a 'vanilla' discriminator cannot take care of local subtleties of
        an image.

        Args:
            in_channels: Number of input channels
            num_filters: Number of filters for the first block.
                Will get doubled for each subsequent block.
            n_down: Number of intermediate blocks

        Example:
            >>> discriminator = PatchDiscriminator(3, n_down=3)
            >>> print(discriminator)
            PatchDiscriminator(
              (model): Sequential(
                (0): Sequential(
                  (0): Conv2d(3, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
                  (1): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (1): Sequential(
                  (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
                  (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (2): Sequential(
                  (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
                  (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (3): Sequential(
                  (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False)
                  (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                  (2): LeakyReLU(negative_slope=0.2, inplace=True)
                )
                (4): Sequential(
                  (0): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
                )
              )
            )
            >>> dummy_input = torch.randn(16, 3, 256, 256)  # batch_size, channels, size, size
            >>> dummy_out = discriminator(dummy_input)
            >>> print(dummy_out.shape)  # batch_size, 1, output_x_shape, output_y_shape
            torch.Size([16, 1, 30, 30])
        """
        super().__init__()

        out_channels = num_filters
        # First layer, w/o normalization
        model = [self.get_block(in_channels, out_channels, norm=False)]
        # Middle layers
        for i in range(n_down):
            in_channels = out_channels
            out_channels *= 2
            stride = 1 if i == n_down - 1 else 2
            model += [self.get_block(in_channels, out_channels, stride=stride)]
        # Last layer, w/o normalization and w/o activation
        model += [self.get_block(out_channels, 1, stride=1, norm=False, activation=False)]
        self.model = nn.Sequential(*model)

    @staticmethod
    def get_block(in_channels: int, out_channels: int, kernel_size=4,
                  stride=2, padding=1, norm=True, activation=True):
        """
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution. Default: 1
            padding: Padding added to all four sides of the input. Default: 0
            norm: Whether to add normalization (skip on first and last layer)
            activation: Whether to add activation (skip on last layer)

        Returns:
            Conv-BatchNorm-LeakyReLU layers
        """
        use_bias = not norm  # add bias only on first and last block
        block = [nn.Conv2d(in_channels, out_channels, kernel_size,
                           stride, padding, bias=use_bias)]
        block += [nn.BatchNorm2d(out_channels)] if norm else []
        block += [nn.LeakyReLU(0.2, True)] if activation else []
        return nn.Sequential(*block)

    def forward(self, x):
        return self.model(x)
