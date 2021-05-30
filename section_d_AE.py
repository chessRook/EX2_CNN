from collections import namedtuple

from torchvision.models import vgg

import main
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import torchvision.models as models
import torchvision.transforms as transform

##################################
writer = SummaryWriter()
##################################
latent_space = 1_00
device = 'cuda'
data_loader = main.data_loader
##################################
encoder = main.Encoder(resolution=(28, 28), channels=1, compression_size=latent_space)
decoder = main.Decoder(channels=1, out_resolution=(28, 28), latent_size=latent_space)

CLASSIFIER_PATH = Path('./Trained_GANS_4_INFERENCE_dont_touch/'
                       'mnist_classifier_encoder_d_part')
classifier = main.Encoder(resolution=(28, 28), channels=1, compression_size=10)
classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=device))
classifier.to('cuda')

epochs = 10


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

enc_optim = torch.optim.Adam(encoder.parameters(), lr=.000009)
dec_optim = torch.optim.Adam(decoder.parameters(), lr=.000009)
global_optim = torch.optim.Adam([{'params': encoder.parameters()},
                                 {'params': decoder.parameters()}], lr=.000009)

####################################

ENC_PATH = Path('./section_d_encoder_part')
DEC_PATH = Path('./section_d_decoder_part')


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

    vector = img2c.view(batch_size, -1)  # image 2 vec
    out = classifier_.fc1(vector)
    out_normalizer = classifier_.final_activation(out)
    features = (img2a, img2b, img2c, out_normalizer)
    return features


#######################################
def trainer():
    stepper = iter(range(int(1e10)))
    for epoch in range(epochs):
        for (images, labels) in data_loader:
            idx = stepper.__next__()
            if idx % 3_000 == 0:
                shower(images)
            images.to(device)
            latent_var = encoder(images)
            out_images = decoder(latent_var)
            loss = perceptual_loss(classifier, out_images, images)
            loss.backward()
            global_optim.step()
            global_optim.zero_grad()
            writer.add_scalar('section_D_LOSS', loss.item(), idx)
            if idx % 2_000 == 1_999:
                save_model()


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
    trainer()
