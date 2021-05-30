import section_3
import matplotlib.pyplot as plt
import main
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import torchvision.transforms as transform
import mnist_classifier
import torch.nn as nn
import numpy as np
import random


def trained_loader():
    encoder_ = main.Encoder(resolution=(28, 28), channels=1, compression_size=latent_space)
    decoder_ = main.Decoder(channels=1, out_resolution=(28, 28), latent_size=latent_space)
    ##############################################################################################
    encoder_.load_state_dict(torch.load(ENC_PATH, map_location=device))
    decoder_.load_state_dict(torch.load(DEC_PATH, map_location=device))
    decoder_.to(device)
    encoder_.to(device)
    return encoder_, decoder_


##################################
writer = SummaryWriter()
##################################
latent_space = 1_00
device = 'cuda'
main.batch_size = 20
data_loader = torch.utils.data.DataLoader(main.mnist_data, batch_size=main.batch_size)
##################################
encoder = main.Encoder(resolution=(28, 28), channels=1, compression_size=latent_space)
decoder = main.Decoder(channels=1, out_resolution=(28, 28), latent_size=latent_space)
########################################################################################


ENC_PATH = Path('Trained_GANS_4_INFERENCE_dont_touch/section_e_encoder_part')
DEC_PATH = Path('Trained_GANS_4_INFERENCE_dont_touch/section_e_decoder_part')
####################################
#
# encoder, decoder = trained_loader()
#######################################################################################
epochs = 9

##########################

decoder.to(device)
encoder.to(device)

##################################
global_optim = torch.optim.Adam([{'params': encoder.parameters()},
                                 {'params': decoder.parameters()}], lr=.001)


def trainer():
    stepper = iter(range(int(1e10)))
    for epoch in range(epochs):
        for (images, labels) in data_loader:
            idx = stepper.__next__()
            if idx % 5_000 == 4_999:
                shower(images)
            if idx % 6_001 == 6_000:
                generate_experiment()

            images.to(device)
            latent_var = encoder(images)
            out_images = decoder(latent_var)
            loss, reconstruction_loss, gaussian_loss_ = loss_method(out_images, images, latent_var)
            loss.backward()
            global_optim.step()
            global_optim.zero_grad()
            writer.add_scalar('section_E_LOSS', loss.item(), idx)
            writer.add_scalar('section_E_Reconstruction_LOSS', reconstruction_loss.item(), idx)
            writer.add_scalar('section_E_Gaussian_LOSS', gaussian_loss_.item(), idx)
            if idx % 2_000 == 1_999:
                save_model()


def loss_method(img1, img2, z):
    img1, img2 = img1.to(device), img2.to(device)
    loss_1 = nn.L1Loss()(img1, img2)
    loss_2 = gaussian_loss(z)
    loss = .75 * loss_1 + .25 * loss_2
    return loss, loss_1, loss_2


def gaussian_loss(z):
    avgs = torch.mean(z, dim=-1)
    variance = torch.var(z, dim=-1)
    avgs_long = torch.tensor([avgs.tolist(), ] * 1_00).to(device).transpose(0, 1)
    kurtosis = torch.mean((z - avgs_long) ** 4, dim=-1)
    l2 = torch.linalg.norm
    loss = l2(avgs) + l2(variance - 1) + l2(kurtosis - 3)
    loss /= main.batch_size
    return loss


def shower(images):
    pred_images = decoder(encoder(images))
    img, pred_img = images[0], pred_images[0]
    img_, pred_img_ = transform.ToPILImage()(img), transform.ToPILImage()(pred_img)
    img_.show()
    pred_img_.show()


def save_model():
    torch.save(encoder.state_dict(), ENC_PATH)
    torch.save(decoder.state_dict(), DEC_PATH)


def gaussian_z_sampler():
    z = torch.tensor([np.random.multivariate_normal(mean=np.zeros(shape=(main.batch_size * latent_space,)),
                                                    cov=np.eye(main.batch_size * latent_space, ),
                                                    size=(1,))])
    z_shaped = z.reshape(main.batch_size, latent_space).float()
    return z_shaped


def generate_experiment():
    __, decoder_ = trained_loader()
    z = gaussian_z_sampler()
    images = decoder_(z)
    for i in range(main.batch_size):
        plt.imshow(images[i, 0].cpu().detach().numpy())
        plt.title(f'Gaussian AE Generator experiment : img_num={i}')
        plt.show()


if __name__ == '__main__':
    trainer()
