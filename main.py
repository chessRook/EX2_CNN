import torch, torchvision
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm

######################### HELPERS ###########################
def int_to_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def conv_out_shape(h_w, kernel_size=3, stride=2, pad=0):
    h_w, kernel_size, stride, pad = int_to_tuple(h_w), \
                                    int_to_tuple(kernel_size), int_to_tuple(stride), int_to_tuple(pad)

    h = np.floor((h_w[0] + (2 * pad[0]) - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int)
    w = np.floor((h_w[1] + (2 * pad[1]) - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int)

    return h, w


def tconv_out_shape(h_w, kernel_size=3, stride=1, pad=0):
    h_w, kernel_size, stride, pad = int_to_tuple(h_w), \
                                    int_to_tuple(kernel_size), int_to_tuple(stride), int_to_tuple(pad)

    h = (h_w[0] - 1) * stride[0] - pad[0] + kernel_size[0]
    w = (h_w[1] - 1) * stride[1] - pad[1] + kernel_size[1]

    return h, w


###################### CODE #######################

writer = SummaryWriter()
device = 'cuda'


class Encoder(nn.Module):
    def __init__(self, resolution=(28, 28), channels=1, compression_size=20):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, (3, 3), stride=(2, 2))
        self.conv2 = nn.Conv2d(self.conv1.out_channels, self.conv1.out_channels * 2, (3, 3), stride=(2, 2))
        self.conv3 = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels * 2, (3, 3), stride=(2, 2))
        self.fc1 = nn.Linear(self.conv_width(resolution) ** 2 * self.conv3.out_channels,
                             compression_size)

        self.default_activation = nn.LeakyReLU()
        self.final_activation = nn.Sigmoid()

    @staticmethod
    def conv_width(resolution):
        h = resolution[0]
        for _ in range(3):
            h, _ = conv_out_shape(h, kernel_size=3, stride=2)
        return h

    def forward(self, img):
        batch_size = img.shape[0]
        img = img.to(device)

        img1a = self.conv1(img)
        img2a = self.default_activation(img1a)

        img1b = self.conv2(img2a)
        img2b = self.default_activation(img1b)

        img1c = self.conv3(img2b)
        img2c = self.default_activation(img1c)

        vector = img2c.view(batch_size, -1)  # image 2 vec
        out = self.fc1(vector)
        out_normalizer = self.final_activation(out)

        return out_normalizer


class Decoder(nn.Module):
    def __init__(self, channels=1, out_resolution=(8, 8), input_size=20):
        super(Decoder, self).__init__()
        self.out_resolution = out_resolution
        self.input_size = input_size
        self.deConv1 = nn.ConvTranspose2d(in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(2, 2),
                                          padding=(3, 3))
        self.deConv2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4), stride=(2, 2),
                                          padding=(1, 1))
        self.deConv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=(2, 2))
        self.deConv4 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(4, 4), stride=(2, 2))
        self.default_activation = nn.LeakyReLU()
        self.final_activation = nn.Sigmoid()
        self.img_width = 28

    def forward(self, z):
        z = z.to(device)
        img1a = self.deConv1(z)
        img1b = self.default_activation(img1a)

        img2a = self.deConv2(img1b)
        img2b = self.default_activation(img2a)

        img3a = self.deConv3(img2b)
        img3b = self.default_activation(img3a)

        img4a = self.deConv4(img3b)
        img4b = self.default_activation(img4a)

        result = self.final_activation(img4b)
        return result



@staticmethod
def __closest_sqr(num):
    sqrt_num = np.sqrt(num)
    num_ = int(sqrt_num) + 1
    return num_


loss_1 = lambda x, y: -nn.BCELoss()(x, y)


def loss_2():
    return 0


def loss_3():
    return 0


######################################################################
discriminator = Encoder(resolution=(28, 28), compression_size=1).to(device)
generator = Decoder(channels=1, out_resolution=(28, 28), input_size=20).to(device)
#######################################################################
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=.001)
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=.001)
#######################################################################
transform = torchvision.transforms.Compose([transforms.ToTensor(), ])
mnist_data = torchvision.datasets.MNIST(download=True, root='./', transform=transform)
data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=5)
#######################################################################
counter = iter(range(int(1e10)))
batch_size = 5
######################################################################
fake_label = torch.zeros(size=(batch_size, 1)).to(device)
real_label = torch.ones(size=(batch_size, 1)).to(device)
######################################################################
noise_size = 4


######################################################################


def gan_trainer(epochs, k, loss):
    for epoch in range(epochs):
        for idx, (img, img_digs) in tqdm(enumerate(data_loader)):
            z = torch.randn(size=(batch_size, 1, noise_size, noise_size))
            ##################################################
            for inner_idx in range(k):
                loss_disc = train_discriminator(generator, discriminator, z, img, loss, disc_optimizer)
                writer.add_scalar('disc_losses', loss_disc, counter.__next__())
            loss_gen = train_generator(generator, discriminator, z, loss, gen_optimizer)
            writer.add_scalar('gen_losses', loss_gen, counter.__next__())
            ###################################################
            if idx % 350 == 1:
                show_results(generator, img)


def show_results(generator, img):
    z = torch.randn(size=(1, 1, noise_size, noise_size))
    tensor_image = generator(z).squeeze(0)
    image = torchvision.transforms.ToPILImage()(tensor_image)
    image.show()
    print(image)
    ###############################
    torchvision.transforms.ToPILImage()(img[0]).show()


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
    loss_ = loss(pred, fake_label)  # log(1 - D(G(z))) want to minimize
    optim_step(loss_, optimizer)
    return loss_.item()


def train_discriminator(generator, discriminator, z, real_img, loss, optimizer):
    pred_for_real_img = discriminator(real_img)
    loss_on_real = -loss(pred_for_real_img, real_label)  # log(D(x)) --> want to maximize the loss.
    optim_step(loss_on_real, optimizer)
    ###########################################
    fake_img = generator(z)
    pred_for_fake_img = discriminator(fake_img.detach())
    loss_on_fake = -loss(pred_for_fake_img, fake_label)  # log(1 - D(G(z))) --> want to maximize the loss.
    optim_step(loss_on_fake, optimizer)
    ############################################
    return (loss_on_fake + loss_on_real).item()


def optim_step(loss_, optimizer):
    loss_.backward()
    optimizer.step()
    optimizer.zero_grad()


##################################################################


def int_to_tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def conv_out_shape(h_w, kernel_size=3, stride=1, pad=0):
    h_w, kernel_size, stride, pad = int_to_tuple(h_w), \
                                    int_to_tuple(kernel_size), int_to_tuple(stride), int_to_tuple(pad)

    h = np.floor((h_w[0] + (2 * pad[0]) - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int)
    w = np.floor((h_w[1] + (2 * pad[1]) - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int)

    return h, w


def tconv_out_shape(h_w, kernel_size=3, stride=1, pad=0):
    h_w, kernel_size, stride, pad = int_to_tuple(h_w), \
                                    int_to_tuple(kernel_size), int_to_tuple(stride), int_to_tuple(pad)

    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + kernel_size[0] + pad[0]
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + kernel_size[1] + pad[1]

    return h, w


def max_pool_out_shape(h_w, kernel_size=2, stride=2, pad=0):
    h_w, kernel_size, stride, pad = int_to_tuple(h_w), \
                                    int_to_tuple(kernel_size), int_to_tuple(stride), int_to_tuple(pad)

    h = np.floor((h_w[0] + (2 * pad[0]) - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int)
    w = np.floor((h_w[1] + (2 * pad[1]) - (kernel_size[1] - 1) - 1) / stride[0] + 1).astype(int)
    return h, w


def max_unpool_out_shape(h_w, kernel_size=2, stride=2, pad=0):
    h_w, kernel_size, stride, pad = int_to_tuple(h_w), \
                                    int_to_tuple(kernel_size), int_to_tuple(stride), int_to_tuple(pad)

    h = (h_w[0] - 1) * stride[0] - 2 * pad[0] + kernel_size[0]
    w = (h_w[1] - 1) * stride[1] - 2 * pad[1] + kernel_size[1]
    return h, w


##############################################################################


if __name__ == '__main__':
    gan_trainer(3, k=20, loss=loss_1)
