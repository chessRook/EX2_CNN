import torch, torchvision, torchaudio
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
device = 'cuda'


class Encoder(nn.Module):
    def __init__(self, resolution=(28, 28), channels=1, compression_size=20):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, (3, 3))
        self.conv2 = nn.Conv2d(self.conv1.out_channels, self.conv1.out_channels * 2, (3, 3))
        self.conv3 = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels * 2, (3, 3))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(int(resolution[0] / 8 - 7 / 4) * int(resolution[1] / 8 - 7 / 4) * 64,
                             compression_size)
        self.default_activation = nn.ELU()
        self.final_activation = nn.Sigmoid()

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
        vector = img3c.view(-1, 1)  # image 2 vec
        out = self.fc1(vector)
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
        self.un_pool = nn.MaxUnpool2d(kernel_size=(2, 2))
        self.default_activation = nn.ELU()
        self.final_activation = nn.Sigmoid()
        self.img_width = self.__closest_sqr(self.input_size)

    # n --> 2(n-1) --> 2*2*(n-1)
    #
    # f(n) = 4n-4
    # f^3 4n-4-> 4(4n-4)-4 ->  4(4(4n-4)-4)-4 = 4^3*n -4-4^2-4^3 = 64
    #
    # n = [64+4+16+16*4]/16*4
    #
    # 8*8 <-- n

    def pad_me(self, z, final_size):
        w = torch.zeros(final_size).to(device)
        for i in z.shape[0]:
            w[i] = z[i]
        return w

    def forward(self, z):
        if np.sqrt(z.shape[0]) % 1 != 0:
            raise Exception('Assuming square vector!')
        z_padded = self.pad_me(z, self.img_width ** 2)
        square_img = z_padded.view(self.img_width, self.img_width)

        img1a = self.deConv1(square_img)
        img1b = self.default_activation(img1a)
        img1c = self.un_pool(img1b)

        img2a = self.deConv2(img1c)
        img2b = self.default_activation(img2a)
        img2c = self.un_pool(img2b)

        img3a = self.deConv3(img2c)
        img3b = self.default_activation(img3a)
        img3c = self.un_pool(img3b)

        result = self.final_activation(img3c)
        return result

    @staticmethod
    def __closest_sqr(num):
        sqrt_num = np.sqrt(num)
        num_ = int(sqrt_num) + 1
        return num_


def loss_1():
    return 0


def loss_2():
    return 0


def loss_3():
    return 0


def get_images_():
    # TODO mnist generator
    pass


######################################################################
discriminator = Encoder(resolution=(28, 28), compression_size=2)
generator = Decoder(channels=1, out_resolution=(28, 28), input_size=20)
#######################################################################
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=.001)
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=.001)
#######################################################################
mnist_data = torchvision.datasets.MNIST(download=True, root='./')
data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=5)
#######################################################################
counter = iter(range(int(1e10)))


def gan_trainer(epochs, k):
    for epoch in range(epochs):
        for idx, img in enumerate(data_loader):
            z = torch.randn(size=(37,))
            ##################################################
            for inner_idx in range(k):
                loss_disc = train_discriminator(generator, discriminator, z, img, loss_1, disc_optimizer)
                writer.add_scalar('disc_losses', loss_disc, counter.__next__())
            loss_gen = train_generator(generator, discriminator, z, img, gen_optimizer)
            writer.add_scalar('gen_losses', loss_gen, counter.__next__())
            ###################################################


def who_wins(discriminator, generator, get_images, threshold):
    with torch.no_grad():
        pred = discriminator(generator(get_images.__next__()))
        error_rate = torch.mean(pred)
        if error_rate < threshold:
            return 'discriminator'
        else:
            return 'generator'


def train_generator(generator, discriminator, z, loss, optimizer):
    img = generator(z)
    pred = discriminator(img)
    loss_ = loss(pred, is_real=0)
    optim_step(loss_, optimizer)
    return loss_


def train_discriminator(generator, discriminator, z, real_img, loss, optimizer):
    fake_img = generator(z)
    pred_for_fake_img = discriminator(fake_img)
    pred_for_real_img = discriminator(real_img)
    loss_ = loss(pred_for_fake_img, pred_for_real_img)
    optim_step(loss_, optimizer)
    return loss_


def optim_step(loss_, optimizer):
    loss_.backward()
    optimizer.step()
    optimizer.zero_grad()


if __name__ == '__main__':
    gan_trainer(3, k=2)
