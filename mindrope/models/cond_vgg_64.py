import torch
import torch.nn as nn


class CondConv2d(nn.Module):
    def __init__(
        self, cond_dim, in_channels, out_channels, kernel_size, stride=1, padding=0
    ):
        super(CondConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
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
        self, cond_dim, in_channels, out_channels, kernel_size, stride=1, padding=0
    ):
        super(CondConvTranspose2d, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
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
        self.c1_1 = CondVGGLayer(ncond, 3, 64)
        self.c1_2 = CondVGGLayer(ncond, 64, 64)

        # 32 x 32
        self.c2_1 = CondVGGLayer(ncond, 64, 128)
        self.c2_2 = CondVGGLayer(ncond, 128, 128)

        # 16 x 16
        self.c3_1 = CondVGGLayer(ncond, 128, 256)
        self.c3_2 = CondVGGLayer(ncond, 256, 256)
        self.c3_3 = CondVGGLayer(ncond, 256, 256)

        # 8 x 8
        self.c4_1 = CondVGGLayer(ncond, 256, 512)
        self.c4_2 = CondVGGLayer(ncond, 512, 512)
        self.c4_3 = CondVGGLayer(ncond, 512, 512)

        # 4 x 4
        self.c5 = nn.Sequential(
            CondConv2d(ncond, 512, dim, 4, 1, 0),
            nn.BatchNorm2d(dim),
            nn.Tanh()
        )
        self.mp = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, input, cond):
        h1_1 = self.c1_1((input, cond))
        h1_2 = self.c1_2((h1_1, cond))
        h1 = self.mp(h1_2)

        h2_1 = self.c2_1((h1, cond))
        h2_2 = self.c2_2((h2_1, cond))
        h2 = self.mp(h2_2)

        h3_1 = self.c3_1((h2, cond))
        h3_2 = self.c3_2((h3_1, cond))
        h3_3 = self.c3_3((h3_2, cond))
        h3 = self.mp(h3_3)

        h4_1 = self.c4_1((h3, cond))
        h4_2 = self.c4_2((h4_1, cond))
        h4_3 = self.c4_3((h4_2, cond))
        h4 = self.mp(h4_3)

        h5 = self.c5((h4, cond))
        return h5.view(-1, self.dim), [h1_2, h2_2, h3_3, h4_3]


class CondVGGDecoder(nn.Module):
    def __init__(self, dim, nskip=1, ncond=7):
        super(CondVGGDecoder, self).__init__()
        self.dim = dim
        self.nskip = nskip + 1
        self.ncond = ncond
        # 1 x 1 -> 4 x 4
        self.upc1 = nn.Sequential(
            CondConvTranspose2d(ncond, dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # 8 x 8
        self.upc2_1 = CondVGGLayer(ncond, 512 * self.nskip, 512)
        self.upc2_2 = CondVGGLayer(ncond, 512, 512)
        self.upc2_3 = CondVGGLayer(ncond, 512, 256)

        # 16 x 16
        self.upc3_1 = CondVGGLayer(ncond, 256 * self.nskip, 256)
        self.upc3_2 = CondVGGLayer(ncond, 256, 256)
        self.upc3_3 = CondVGGLayer(ncond, 256, 128)

        # 32 x 32
        self.upc4_1 = CondVGGLayer(ncond, 128 * self.nskip, 128)
        self.upc4_2 = CondVGGLayer(ncond, 128, 64)

        # 64 x 64
        self.upc5 = CondVGGLayer(ncond, 64 * self.nskip, 64)

        self.upc6 = nn.Sequential(
            CondConvTranspose2d(ncond, 64, 3, 3, 1, 1),
            nn.Sigmoid()
        )
        self.up = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, input, cond):
        vec, skip = input
        d1 = self.upc1((vec.view(-1, self.dim, 1, 1), cond))  # 1 -> 4
        up1 = self.up(d1)  # 4 -> 8

        d2_1 = self.upc2_1((torch.cat([up1, skip[3]], 1), cond))
        d2_2 = self.upc2_2((d2_1, cond))
        d2_3 = self.upc2_3((d2_2, cond))
        up2 = self.up(d2_3)  # 8 -> 16

        d3_1 = self.upc3_1((torch.cat([up2, skip[2]], 1), cond))
        d3_2 = self.upc3_2((d3_1, cond))
        d3_3 = self.upc3_3((d3_2, cond))
        up3 = self.up(d3_3)  # 16 -> 32

        d4_1 = self.upc4_1((torch.cat([up3, skip[1]], 1), cond))
        d4_2 = self.upc4_2((d4_1, cond))
        up4 = self.up(d4_2)  # 32 -> 64

        d5 = self.upc5((torch.cat([up4, skip[0]], 1), cond))
        output = self.upc6((d5, cond))

        return output
