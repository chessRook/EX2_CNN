import torch, torchvision, torchaudio
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self):
        self.


class Encoder(nn.Module):
    def __init__(self, channels=1, resolution=(8,8)):
        super(Encoder, self).__init__()
        self.resolution = resolution
        self.conv1 = nn.Conv2d(channels, 16, (3,3))
        self.conv1 = nn.Conv2d(self.conv1.out_channels, self.conv1.out_channels*2, (3,3))
        self.conv1 = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels*2, (3,3))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(m/8-7/4, 64)
        self.fc1 = nn.Linear()


n*n*3 -> m-2*m-2*16 -> m/2-1*m/2-1*16 -> m2-3*32 --> m/4-1.5*32 -> m/4-3.5*64 -> m/8-7/4*64

        conv -> pool -> conv -> pool ->conv -> pool ->fc

def main():


if __name__ == '__main__':
    main()
