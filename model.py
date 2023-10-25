import torch.nn as nn
from torch.nn import Module
import torch
from torchvision.transforms import CenterCrop


class Block(Module):
    def __init__(self, _in, _out):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(_in, _out, 3)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(_out, _out, 3)

    def forward(self, X):
        return self.conv2(self.relu(self.conv1(X)))


class Encoder(Module):
    def __init__(self, channels=(3,16,32,64)):
        super(Encoder, self).__init__()

        self.enc = nn.ModuleList(
            [Block(channels[i], channels[i+1])
                for i in range(len(channels)-1)])
        self.pool = nn.MaxPool2d(2)

    def forward(self, X):
        outputs = []

        for block in self.enc:
            X = block(X)
            outputs.append(X)
            X = self.pool(X)

        return outputs


class Decoder(Module):
    def __init__(self, channels=(64,32,16)):
        super(Decoder, self).__init__()

        self.channels = channels
        self.upconvs = nn.ModuleList(
            [nn.ConvTranspose2d(channels[i], channels[i+1], 2, 2)
                for i in range(len(channels)-1)]
        )
        self.dec_blocks = nn.ModuleList(
            [Block(channels[i], channels[i+1])
                for i in range(len(channels)-1)]
        )

    def forward(self, x, encFeatures):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)

        return x

    def crop(self, encFeatures, x):
        (_,_,H,W) = x.shape
        encFeatures = CenterCrop([H,W])(encFeatures)

        return encFeatures


class UNET(Module):
    def __init__(self, encChannels=(3,16,32,64),
                decChannels=(64,32,16), nbClasses=3,
                retainDim=True, outsize=(512,512)):
        super(UNET, self).__init__()

        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)

        self.head = nn.Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outsize

    def forward(self, x):
        encFeatures = self.encoder(x)

        decFeatures = self.decoder(encFeatures[::-1][0],
                                   encFeatures[::-1][1:])

        map = self.head(decFeatures)

        if self.retainDim:
            map = nn.functional.interpolate(map, self.outSize)

        return map