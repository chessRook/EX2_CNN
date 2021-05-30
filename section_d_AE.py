import section_3
import matplotlib.pyplot as plt
import main
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import torchvision.transforms as transform
import mnist_classifier
import torch.nn as nn

##################################
writer = SummaryWriter()
##################################
latent_space = 1_00
device = 'cuda'
data_loader = main.data_loader
##################################
encoder = main.Encoder(resolution=(28, 28), channels=1, compression_size=latent_space)
decoder = main.Decoder(channels=1, out_resolution=(28, 28), latent_size=latent_space)

#######################################################################################

CLASSIFIER_PATH = Path('./Trained_GANS_4_INFERENCE_dont_touch/'
                       'mnist_classifier_encoder_d_part')
classifier = mnist_classifier.BigEncoder(resolution=(28, 28), channels=1, compression_size=10)
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
classifier.to('cuda')

##########################################################################################
mnist_classifier.tester(classifier)
mnist_classifier.tester(classifier)
mnist_classifier.tester(classifier)
##########################################################################################
epochs = 10


##########################

def freezer(model):
    model.train(False)
    classifier.eval()
    for p in model.parameters():
        p.requires_grad = False


freezer(classifier)

##################################

decoder.to(device)
encoder.to(device)

##################################
global_optim = torch.optim.Adam([{'params': encoder.parameters()},
                                 {'params': decoder.parameters()}], lr=.001)

####################################

ENC_PATH = Path('section_d_good_results/section_d_encoder_part')
DEC_PATH = Path('section_d_good_results/section_d_decoder_part')


def trainer():
    stepper = iter(range(int(1e10)))
    for epoch in range(epochs):
        for (images, labels) in data_loader:
            idx = stepper.__next__()
            if idx % 5_000 == 4_999:
                shower(images)
            if idx % 10_000 == 9_999:
                experiment_ae()
            images.to(device)
            latent_var = encoder(images)
            out_images = decoder(latent_var)
            # loss = perceptual_loss(classifier, out_images, images)
            # loss = nn.MSELoss()(out_images.to(device), images.to(device))
            loss = nn.L1Loss()(out_images.to(device), images.to(device))
            loss.backward()
            global_optim.step()
            global_optim.zero_grad()
            writer.add_scalar('section_D_LOSS', loss.item(), idx)
            if idx % 2_000 == 1_999:
                save_model()


def experiment_ae():
    z_s_lst = []
    for i, (images, labels) in enumerate(data_loader):
        z_s = encoder(images)
        z_s_lst.append(z_s)
        if i >= 2:
            break

    z_0 = z_s_lst[0]
    z_1 = z_s_lst[1]

    for j in range(11):
        z_j = (j / 10) * z_0 + (1 - j / 10) * z_1
        img_j = decoder(z_j)
        ###########################################
        plt.imshow(img_j[0, 0].cpu().detach().numpy())
        plt.title(f'AE interpolation experiment : alpha={j / 10}')
        plt.show()
        # ###########################################
    plt.show()


def experiment_with_gan_section_d():
    z_s_lst = []
    for i, (images, labels) in enumerate(data_loader):
        __, z_s = section_3.z_searcher(images, images)
        z_s_lst.append(z_s)
        if i >= 2:
            break

    z_0 = z_s_lst[0]
    z_1 = z_s_lst[1]

    for j in range(11):
        z_j = (j / 10) * z_0 + (1 - j / 10) * z_1
        img_j = section_3.trained_generator(z_j)
        ###########################################
        plt.imshow(img_j[0, 0].cpu().detach().numpy())
        plt.title(f'GAN INTERPOLATION , alpha={j / 10}')
        plt.show()
        # ###########################################
    plt.show()


def shower(images):
    pred_images = decoder(encoder(images))
    img, pred_img = images[0], pred_images[0]
    img_, pred_img_ = transform.ToPILImage()(img), transform.ToPILImage()(pred_img)
    img_.show()
    pred_img_.show()


def save_model():
    torch.save(encoder.state_dict(), ENC_PATH)
    torch.save(decoder.state_dict(), DEC_PATH)


if __name__ == '__main__':
    experiment_with_gan_section_d()
    trainer()


####################################
####################################
####################################


def perceptual_loss(model, img1, img2):
    img1_features = features_extractor(model, img1)
    img2_features = features_extractor(model, img2)
    loss = 0
    for img1_feature, img2_feature in zip(img1_features, img2_features):
        loss += torch.nn.MSELoss()(img1_feature, img2_feature)
    return loss


def features_extractor(classifier_, img):
    batch_size = img.shape[0]
    img = img.to(device)

    img1a = classifier_.conv1(img)
    img2a = classifier_.default_activation(classifier_.bn1(img1a))

    img1b = classifier_.conv2(img2a)
    img2b = classifier_.default_activation(classifier_.bn2(img1b))

    img1c = classifier_.conv3(img2b)
    img2c = classifier_.default_activation(img1c)

    img1d = classifier_.conv4(img2c)
    img2d = classifier_.default_activation(img1d)

    img1e = classifier_.conv5(img2d)
    img2e = classifier_.default_activation(img1e)

    vector = img2e.view(batch_size, -1)  # image 2 vec
    out = classifier_.fc1(vector)
    out_normalizer = classifier_.final_activation(out)

    features = (img2e,)
    return features

#######################################
