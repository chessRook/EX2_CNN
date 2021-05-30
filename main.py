import torch, torchvision
import torch.nn as nn
import numpy as np
from pandas._libs.parsers import k
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import atexit


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
        self.conv1 = nn.Conv2d(channels, 16, (4, 4), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, self.conv1.out_channels * 2, (4, 4), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3 = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels * 2, (4, 4), stride=(2, 2))
        self.fc1 = nn.Linear(self.conv_width(resolution) ** 2 * self.conv3.out_channels,
                             compression_size)

        self.default_activation = nn.LeakyReLU()
        self.final_activation = nn.Sigmoid()

    @staticmethod
    def conv_width(resolution):
        h = resolution[0]
        for _ in range(3):
            h, _ = conv_out_shape(h, kernel_size=4, stride=2)
        return h

    def forward(self, img):
        batch_size = img.shape[0]
        img = img.to(device)

        img1a = self.conv1(img)
        img2a = self.default_activation(self.bn1(img1a))

        img1b = self.conv2(img2a)
        img2b = self.default_activation(self.bn2(img1b))

        img1c = self.conv3(img2b)
        img2c = self.default_activation(img1c)

        vector = img2c.view(batch_size, -1)  # image 2 vec
        out = self.fc1(vector)
        out_normalizer = self.final_activation(out)

        return out_normalizer


class Decoder(nn.Module):
    def __init__(self, channels=1, out_resolution=(8, 8), latent_size=20):
        super(Decoder, self).__init__()
        self.out_resolution = out_resolution
        self.latent_size = latent_size
        self.fc1 = nn.Linear(latent_size, 4 * 4 * latent_size)  # 4 is to make out resolution good
        self.tConv2 = nn.ConvTranspose2d(in_channels=100, out_channels=32, kernel_size=(4, 4), stride=(2, 2),
                                         padding=(2, 2))
        self.tConv3 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(4, 4), stride=(2, 2))
        self.tConv4 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(4, 4), stride=(2, 2),
                                         padding=(1, 1))

        self.bn1 = nn.BatchNorm2d(int(self.fc1.out_features / 16))
        self.bn2 = nn.BatchNorm2d(self.tConv2.out_channels)
        self.bn3 = nn.BatchNorm2d(self.tConv3.out_channels)

        self.default_activation = nn.LeakyReLU()
        self.final_activation = nn.Sigmoid()
        self.img_width = 28

    def forward(self, z):
        z = z.squeeze().to(device)
        img1a = self.fc1(z)
        img1a_viewed = img1a.view(batch_size, -1, 4, 4)
        img1b = self.default_activation(self.bn1(img1a_viewed))

        img2a = self.tConv2(img1b)
        img2b = self.default_activation(self.bn2(img2a))

        img3a = self.tConv3(img2b)
        img3b = self.default_activation(self.bn3(img3a))

        img4a = self.tConv4(img3b)
        img4b = self.default_activation(img4a)

        result = self.final_activation(img4b)
        return result


    @staticmethod
    def __closest_sqr(num):
        sqrt_num = np.sqrt(num)
        num_ = int(sqrt_num) + 1
        return num_


mini_max_loss_gen = lambda x, y: -nn.BCELoss()(x, y)
mini_max_loss_disc = lambda x, y: nn.BCELoss()(x, y)

non_staurating_gen_loss = lambda pred, _: (-.5 * torch.log(pred)).mean()

ls_loss_disc = lambda pred, label: (.5 * (pred - label) ** 2).mean()
ls_loss_gen = lambda pred, _: (.5 * (pred - real_label) ** 2).mean()

######################################################################
noise_size = 1_00
######################################################################
discriminator = Encoder(resolution=(28, 28), compression_size=1).to(device)
generator = Decoder(channels=1, out_resolution=(28, 28), latent_size=noise_size).to(device)
#######################################################################
lr = .0007
disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr)
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)
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

PATH_TO_SAVE_DISC = Path('./Trained_GANS_4_INFERENCE/trained_discriminator')
PATH_TO_SAVE_GEN = Path('./Trained_GANS_4_INFERENCE/trained_generator')


######################################################################
class List50:
    def __init__(self):
        self.last50 = []

    def get_avg(self):
        if self.last50.__len__() <= 50:
            return None
        else:
            return sum(self.last50) / 50

    def add_number(self, num):
        if self.last50.__len__() >= 50:
            self.last50.pop(0)
        self.last50.append(num)


def gan_trainer(epochs, k_disc, k_gen, loss_gen_method, loss_disc_method):
    disc_losses_50, gen_losses_50 = List50(), List50()
    for epoch in range(epochs):
        for idx, (img, img_digs) in tqdm(enumerate(data_loader)):
            ##################################################
            for inner_idx in range(k_disc):
                z = get_z(noise_size=100)
                loss_disc_on_fake, loss_disc_on_real = train_discriminator(generator, discriminator, z, img,
                                                                           loss_disc_method, disc_optimizer)
                writer.add_scalar('disc_losses_on_fake', loss_disc_on_fake, counter.__next__())
                writer.add_scalar('disc_losses_on_real', loss_disc_on_real, counter.__next__())
                disc_losses_50.add_number(loss_disc_on_fake + loss_disc_on_real)
            ##########################################################
            for inner_idx in range(k_gen):
                z = get_z(noise_size=100)
                loss_gen = train_generator(generator, discriminator, z, loss_gen_method, gen_optimizer)
                writer.add_scalar('gen_losses', loss_gen, counter.__next__())
                gen_losses_50.add_number(loss_gen)
            ###################################################
            if idx % 3_000 == 150:
                show_results(generator, img)
                model_saver()
            if idx % 2_000 == 1999:
                k_gen, k_disc = who_wins(disc_losses_50, gen_losses_50, k_gen, k_disc)
                writer.add_scalar('k_disc', k_disc, int(idx / 2_000) + 1)
                writer.add_scalar('k_gen', k_gen, int(idx / 2_000) + 1)


def who_wins(disc_losses_50, gen_losses_50, k_gen, k_disc):
    disc_avg = disc_losses_50.get_avg()
    gen_avg = gen_losses_50.get_avg()
    if disc_avg is None or gen_avg is None:
        return k_gen, k_disc
    if disc_avg < .08 and gen_avg > .2:
        if k_disc > 1:
            k_disc -= 1
        else:
            k_gen += 1
    if disc_avg > .2 and gen_avg < .08:
        if k_gen > 1:
            k_gen -= 1
        else:
            k_disc += 1
    return min(k_gen, 4), min(k_disc, 4)


def get_z(noise_size=100):
    z = torch.randn(size=(batch_size, noise_size, 1, 1)).to(device)
    return z


def show_results(generator, img):
    torchvision.transforms.ToPILImage()(img[0]).show()
    ############################################################
    z = get_z(100)
    tensor_image = generator(z).squeeze(0)
    image = torchvision.transforms.ToPILImage()(tensor_image[0])
    image.show()
    print(image)


def train_generator(generator, discriminator, z, loss, optimizer):
    img = generator(z)
    pred = discriminator(img)
    loss_ = loss(pred, fake_label)  # log(1 - D(G(z))) want to minimize WANT --- -INF
    optim_step(loss_, optimizer)
    return loss_.item()


def train_discriminator(generator, discriminator, z, real_img, loss, optimizer):
    pred_for_real_img = discriminator(real_img)
    loss_on_real = loss(pred_for_real_img, real_label)  # log(D(x)) --> want to maximize the loss. --- WANT 0
    optim_step(loss_on_real, optimizer)
    ###########################################
    fake_img = generator(z)
    pred_for_fake_img = discriminator(fake_img.detach())
    loss_on_fake = loss(pred_for_fake_img, fake_label)  # log(1 - D(G(z))) --> want to maximize the loss.  WANT 0
    optim_step(loss_on_fake, optimizer)
    ############################################
    return loss_on_fake.item(), loss_on_real.item()


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


def model_saver():
    torch.save(discriminator.state_dict(), PATH_TO_SAVE_DISC)
    torch.save(generator.state_dict(), PATH_TO_SAVE_GEN)





##############################################################################
def exit_handler():
    model_saver()


atexit.register(exit_handler)

##############################################################################


if __name__ == '__main__':
    gan_trainer(epochs=3, k_disc=1, k_gen=1, loss_gen_method=non_staurating_gen_loss, loss_disc_method=mini_max_loss_disc)

###################DOCS###############################
# loss_3 works for k = 1
# loss_2 works for k = ?
# most important to keep both losses far from 0
# in equilibrium such that no one wins
# and for this we tune k !!!
