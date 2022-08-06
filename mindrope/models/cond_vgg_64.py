import torch
import torch.nn as nn


class CondConv2d(nn.Module):
    def __init__(
        self, cond_dim, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
        bias=True
    ):
        super(CondConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.fc_weights = nn.Linear(cond_dim, out_channels)
        self.fc_bias = nn.Linear(cond_dim, out_channels)
        self.softplus = nn.Softplus()

    def forward(self, input):
        input_tensor, cond = input
        weights = self.softplus(self.fc_weights(cond))
        bias = self.fc_bias(cond)
        return self.conv(input_tensor) * weights[:, :, None, None] + bias[:, :, None, None]


class CondConvTranspose2d(nn.Module):
    def __init__(
        self, cond_dim, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
        bias=True
    ):
        super(CondConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.fc_weights = nn.Linear(cond_dim, out_channels)
        self.fc_bias = nn.Linear(cond_dim, out_channels)
        self.softplus = nn.Softplus()

    def forward(self, input):
        input_tensor, cond = input
        weights = self.softplus(self.fc_weights(cond))
        bias = self.fc_bias(cond)
        return self.conv(input_tensor) * weights[:, :, None, None] + bias[:, :, None, None]


class CondVGGLayer(nn.Module):
    def __init__(self, ncond, nin, nout):
        super(CondVGGLayer, self).__init__()
        self.main = nn.Sequential(
            CondConv2d(ncond, nin, nout, 3, 1, 1),
            nn.BatchNorm2d(nout),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.main(input)


class CondVGGEncoder(nn.Module):
    def __init__(self, dim, ncond=7):
        super(CondVGGEncoder, self).__init__()
        self.dim = dim
        self.ncond = ncond
        # 64 x 64
        self.c1 = nn.Sequential(
            CondVGGLayer(ncond, 3, 64),
            CondVGGLayer(ncond, 64, 64),
        )
        # 32 x 32
        self.c2 = nn.Sequential(
            CondVGGLayer(ncond, 64, 128),
            CondVGGLayer(ncond, 128, 128),
        )
        # 16 x 16
        self.c3 = nn.Sequential(
            CondVGGLayer(ncond, 128, 256),
            CondVGGLayer(ncond, 256, 256),
            CondVGGLayer(ncond, 256, 256),
        )
        # 8 x 8
        self.c4 = nn.Sequential(
            CondVGGLayer(ncond, 256, 512),
            CondVGGLayer(ncond, 512, 512),
            CondVGGLayer(ncond, 512, 512),
        )
        # 4 x 4
        self.c5 = nn.Sequential(
            CondConv2d(ncond, 512, dim, 4, 1, 0),
            nn.BatchNorm2d(dim),
            nn.Tanh()
        )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input, cond):
        h1 = self.c1(input)  # 64 -> 32
        h2 = self.c2((self.mp(h1), cond))  # 32 -> 16
        h3 = self.c3((self.mp(h2), cond))  # 16 -> 8
        h4 = self.c4((self.mp(h3), cond))  # 8 -> 4
        h5 = self.c5((self.mp(h4), cond))  # 4 -> 1
        return h5.view(-1, self.dim), [h1, h2, h3, h4]


class CondVGGDecoder(nn.Module):
    def __init__(self, dim, ncond=7):
        super(CondVGGDecoder, self).__init__()
        self.dim = dim
        self.ncond = ncond
        # 1 x 1 -> 4 x 4
        self.upc1 = nn.Sequential(
            CondConvTranspose2d(ncond, dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 8 x 8
        self.upc2 = nn.Sequential(
            CondVGGLayer(ncond, 512*2, 512),
            CondVGGLayer(ncond, 512, 512),
            CondVGGLayer(ncond, 512, 256)
        )
        # 16 x 16
        self.upc3 = nn.Sequential(
            CondVGGLayer(ncond, 256*2, 256),
            CondVGGLayer(ncond, 256, 256),
            CondVGGLayer(ncond, 256, 128)
        )
        # 32 x 32
        self.upc4 = nn.Sequential(
            CondVGGLayer(ncond, 128*2, 128),
            CondVGGLayer(ncond, 128, 64)
        )
        # 64 x 64
        self.upc5 = nn.Sequential(
            CondVGGLayer(ncond, 64*2, 64),
            CondConvTranspose2d(ncond, 64, 3, 3, 1, 1),
            nn.Sigmoid()
        )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input, cond):
        vec, skip = input
        d1 = self.upc1((vec.view(-1, self.dim, 1, 1), cond))  # 1 -> 4
        up1 = self.up(d1)  # 4 -> 8
        d2 = self.upc2((torch.cat([up1, skip[3]], 1), cond))  # 8 x 8
        up2 = self.up(d2)  # 8 -> 16
        d3 = self.upc3((torch.cat([up2, skip[2]], 1), cond))  # 16 x 16
        up3 = self.up(d3)  # 8 -> 32
        d4 = self.upc4((torch.cat([up3, skip[1]], 1), cond))  # 32 x 32
        up4 = self.up(d4)  # 32 -> 64
        output = self.upc5((torch.cat([up4, skip[0]], 1), cond))  # 64 x 64
        return output
