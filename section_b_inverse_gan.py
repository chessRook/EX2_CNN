from main import Encoder
import main
import torch, torchvision
import torchvision.transforms as transforms
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn as nn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import tqdm
#######################################################################
writer = SummaryWriter()

DISC_PATH = Path(r'.\Trained_GANS_4_INFERENCE_dont_touch\trained_discriminator')
GEN_PATH = Path(r'.\Trained_GANS_4_INFERENCE_dont_touch\trained_generator')
INVERSER_PATH = Path(r'.\Trained_GANS_4_INFERENCE_dont_touch\trained_inverser')
device = 'cuda'


#######################################################################


def model_loader():
    discriminator = main.Encoder(resolution=(28, 28), compression_size=1)
    generator = main.Decoder(channels=1, out_resolution=(28, 28), latent_size=main.noise_size)
    ##############################################################################################
    discriminator.load_state_dict(torch.load(DISC_PATH, map_location=device))
    generator.load_state_dict(torch.load(GEN_PATH, map_location=device))

    generator.to(device)
    discriminator.to(device)

    return generator, discriminator


#######################################################################
generator, discriminator = model_loader()
inverser = Encoder(resolution=(28, 28), channels=1, compression_size=generator.latent_size).to('cuda')
inverser_optimizer = torch.optim.Adam(inverser.parameters(), lr=.00005)
training_iterations = int(1e9)


#######################################################################

def inverse_trainer():
    # loss = torch.nn.MSELoss()
    loss = lambda x, y: .7 * torch.nn.L1Loss()(x, y) + .3 * torch.nn.L1Loss()(x, y)
    # loss = nn.SmoothL1Loss()
    for idx in tqdm.tqdm(range(training_iterations)):
        z = main.get_z(generator.latent_size)
        img = generator(z)
        pred_z = inverser(img)
        # loss = ((pred_z - z) ** 2).mean()
        error = loss(pred_z, z.view(pred_z.size()))
        error.backward()
        # loss.backward()
        inverser_optimizer.step()
        inverser_optimizer.zero_grad()
        # main.writer.add_scalar('inverser_loss', loss.item(), idx)
        writer.add_scalar('inverser_loss', error.item(), idx)
        if idx % 3_000 == 0:
            pred_img = generator(pred_z.view(5, 100, 1, 1))
            plt.imshow(pred_img[0, 0].cpu().detach().numpy())
            plt.show()
            plt.imshow(img[0, 0].cpu().detach().numpy())
            plt.show()
        if idx % 9_000 == 0:
            save_section_b_model(inverser)
            plt.close('all')


def img_shower(img):
    for i in range(main.batch_size):
        img_pil = torchvision.transforms.ToPILImage()(img[i])
        img_pil.show()
    sys.exit()


now = datetime.now()
time_sign = f'{now.hour}_{now.minute}_{now.second}'


def save_section_b_model(model):
    path = Path(f'./section_b_inverser_{time_sign}')
    torch.save(model.state_dict(), path)


def inverser_loader():
    inverser_ = Encoder(resolution=(28, 28), channels=1, compression_size=generator.latent_size)
    ##############################################################################################
    inverser_.load_state_dict(torch.load(INVERSER_PATH, map_location=device))
    inverser_.to(device)
    return inverser_


if __name__ == '__main__':
    inverse_trainer()
