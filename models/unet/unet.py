# 2D-UNet model.
import torch
import torch.nn as nn
from models.basic_module import BasicModule


class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

        # first convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class BottomBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(BottomBlock, self).__init__()

        # first convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # residual block
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class DecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # first convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # residual block
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


def conv_trans_block_2d(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size=2, stride=2))


def max_pooling_2d():
    return nn.MaxPool2d(2)


class UNet(BasicModule):
    def __init__(self, in_dim=6, out_dim=1, num_filters=16):
        super(UNet, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters

        # Down sampling
        self.down_1 = EncoderBlock(self.in_dim, self.num_filters)
        self.pool_1 = max_pooling_2d()
        self.down_2 = EncoderBlock(self.num_filters, self.num_filters * 2)
        self.pool_2 = max_pooling_2d()
        self.down_3 = EncoderBlock(self.num_filters * 2, self.num_filters * 4)
        self.pool_3 = max_pooling_2d()
        self.down_4 = EncoderBlock(self.num_filters * 4, self.num_filters * 8)
        self.pool_4 = max_pooling_2d()

        # Bridge
        self.bridge = BottomBlock(self.num_filters * 8, self.num_filters * 16)

        # Up sampling
        self.trans_1 = conv_trans_block_2d(self.num_filters * 16, self.num_filters * 8)
        self.up_1 = DecoderBlock(self.num_filters * 16, self.num_filters * 8)

        self.trans_2 = conv_trans_block_2d(self.num_filters * 8, self.num_filters * 4)
        self.up_2 = DecoderBlock(self.num_filters * 8, self.num_filters * 4)

        self.trans_3 = conv_trans_block_2d(self.num_filters * 4, self.num_filters * 2)
        self.up_3 = DecoderBlock(self.num_filters * 4, self.num_filters * 2)

        self.trans_4 = conv_trans_block_2d(self.num_filters * 2, self.num_filters)
        self.up_4 = DecoderBlock(self.num_filters * 2, self.num_filters)

        # Output
        self.seg_out = nn.Conv2d(self.num_filters, out_dim, 1)

    def forward(self, x):
        # Down sampling
        down_1 = self.down_1(x)
        pool_1 = self.pool_1(down_1)

        down_2 = self.down_2(pool_1)
        pool_2 = self.pool_2(down_2)

        down_3 = self.down_3(pool_2)
        pool_3 = self.pool_3(down_3)

        down_4 = self.down_4(pool_3)
        pool_4 = self.pool_4(down_4)

        # Bridge
        bridge = self.bridge(pool_4)

        # Up sampling
        trans_1 = self.trans_1(bridge)
        concat_1 = torch.cat([trans_1, down_4], dim=1)
        up_1 = self.up_1(concat_1)

        trans_2 = self.trans_2(up_1)
        concat_2 = torch.cat([trans_2, down_3], dim=1)
        up_2 = self.up_2(concat_2)

        trans_3 = self.trans_3(up_2)
        concat_3 = torch.cat([trans_3, down_2], dim=1)
        up_3 = self.up_3(concat_3)

        trans_4 = self.trans_4(up_3)
        concat_4 = torch.cat([trans_4, down_1], dim=1)
        up_4 = self.up_4(concat_4)

        # Output
        seg_out = self.seg_out(up_4)
        seg_out = nn.Sigmoid()(seg_out)

        return seg_out


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.Tensor(1, 6, 1024, 1024).cuda()
    print("x size: {}".format(x.size()))

    model = UNet(in_dim=6, out_dim=1, num_filters=16).to(device)
    out = model(x)
    print(out.size())

