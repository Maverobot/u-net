import mlx.core as mx
import mlx.nn as nn

import mlx
import test_unet_mlx


# from https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/unet.py
def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    x = x.reshape(B, H * scale, W * scale, C)
    return x


class UpsamplingConv2d(nn.Module):
    """
    A convolutional layer that upsamples the input by a factor of 2. MLX does
    not yet support transposed convolutions, so we approximate them with
    nearest neighbor upsampling followed by a convolution. This is similar to
    the approach used in the original U-Net.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def __call__(self, x):
        x = self.conv(upsample_nearest(x))
        return x


class UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.e11 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)  # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)  # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)  # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)  # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)  # output: 28x28x1024

        # Decoder
        self.upconv1 = UpsamplingConv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = UpsamplingConv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = UpsamplingConv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = UpsamplingConv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def __call__(self, x):
        # Encoder
        xe11 = nn.relu(self.e11(x))
        xe12 = nn.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = nn.relu(self.e21(xp1))
        xe22 = nn.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = nn.relu(self.e31(xp2))
        xe32 = nn.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = nn.relu(self.e41(xp3))
        xe42 = nn.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = nn.relu(self.e51(xp4))
        xe52 = nn.relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = mx.concatenate([xu1, xe42], axis=3)
        xd11 = nn.relu(self.d11(xu11))
        xd12 = nn.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = mx.concatenate([xu2, xe32], axis=3)
        xd21 = nn.relu(self.d21(xu22))
        xd22 = nn.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = mx.concatenate([xu3, xe22], axis=3)
        xd31 = nn.relu(self.d31(xu33))
        xd32 = nn.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = mx.concatenate([xu4, xe12], axis=3)
        xd41 = nn.relu(self.d41(xu44))
        xd42 = nn.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out


if __name__ == "__main__":
    test_unet_mlx.run(UNet)
