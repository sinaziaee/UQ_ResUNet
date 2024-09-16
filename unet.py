import torch
import torch.nn as nn

class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.conv_block = UNetConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        skip_connection = x
        x = self.pool(x)
        return x, skip_connection

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_block = UNetConvBlock(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.upconv(x)
        x = torch.cat((x, skip_connection), dim=1)
        x = self.conv_block(x)
        return x

forward_channels1 = [16, 32, 64, 128, 256]
forward_channels2 = [32, 64, 128, 256, 512]
forward_channels3 = [64, 128, 256, 512, 1024]

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, forward_channels=forward_channels3):
        super(UNet, self).__init__()
        self.encoder1 = Encoder(in_channels, forward_channels[0])
        self.encoder2 = Encoder(forward_channels[0], forward_channels[1])
        self.encoder3 = Encoder(forward_channels[1], forward_channels[2])
        self.encoder4 = Encoder(forward_channels[2], forward_channels[3])

        self.bottleneck = UNetConvBlock(forward_channels[3], forward_channels[4])

        self.decoder4 = Decoder(forward_channels[4], forward_channels[3])
        self.decoder3 = Decoder(forward_channels[3], forward_channels[2])
        self.decoder2 = Decoder(forward_channels[2], forward_channels[1])
        self.decoder1 = Decoder(forward_channels[1], forward_channels[0])

        self.final_conv = nn.Conv2d(forward_channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        x = self.bottleneck(x)

        x = self.decoder4(x, skip4)
        x = self.decoder3(x, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)

        x = self.final_conv(x)
        return x
