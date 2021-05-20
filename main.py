import torch, torchvision, torchaudio
import torch.nn as nn
import numpy as np

device = 'cuda'


class Discriminator(nn.Module):
    def __init__(self):
        self.


class Encoder(nn.Module):
    def __init__(self, channels=1, resolution=(8, 8), compression_size=20):
        super(Encoder, self).__init__()
        self.resolution = resolution
        self.conv1 = nn.Conv2d(channels, 16, (3, 3))
        self.conv2 = nn.Conv2d(self.conv1.out_channels, self.conv1.out_channels * 2, (3, 3))
        self.conv3 = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels * 2, (3, 3))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(int(self.resolution[0] / 8 - 7 / 4) * int(self.resolution[1] / 8 - 7 / 4) * 64,
                             compression_size)
        self.default_activation = nn.ELU()
        self.final_activation = nn.Sigmoid()
        # self.optimizer = torch.optim.Adam(params=self.parameters(), lr=.001)

    def forward(self, img):
        img = img.to(device)

        img1a = self.conv1(img)
        img2a = self.default_activation(img1a)
        img3a = self.pool(img2a)

        img1b = self.conv2(img3a)
        img2b = self.default_activation(img1b)
        img3b = self.pool(img2b)

        img1c = self.conv3(img3b)
        img2c = self.default_activation(img1c)
        img3c = self.pool(img2c)

        out = self.fc1(img3c)
        out_normalizer = self.final_activation(out)

        return out_normalizer


class Decoder(nn.Module):
    def __init__(self, channels=1, out_resolution=(8, 8), input_size=20):
        super(Decoder, self).__init__()
        self.out_resolution = out_resolution
        self.input_size = input_size
        self.deConv1 = nn.ConvTranspose2d(in_channels=1, out_channels=16, kernel_size=(3, 3))
        self.deConv2 = nn.ConvTranspose2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.deConv3 = nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
        self.pool = nn.MaxUnpool2d(kernel_size=(2,2))
        self.default_activation = nn.ELU()
        self.final_activation = nn.Sigmoid()
        self.img_width = self.__closest_sqr(self.input_size)

    def forward(self, comp):
        comp_wider = torch.cat()
        torch.reshape(comp, shape=(self.img_width, self.img_width))

        img1 = self.conv1(img)
        img2 = self.deConv3



    @staticmethod
    def __closest_sqr(num):
        sqrt_num = np.sqrt(num)
        num_ = int(sqrt_num) + 1
        return num_


#
# n * n * 3 -> m - 2 * m - 2 * 16 -> m / 2 - 1 * m / 2 - 1 * 16 -> m2 - 3 * 32 --> m / 4 - 1.5 * 32 -> m / 4 - 3.5 * 64 -> m / 8 - 7 / 4 * 64
#
# conv -> pool -> conv -> pool ->conv -> pool ->fc
#

def main():


if __name__ == '__main__':
    main()
