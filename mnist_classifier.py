from collections import namedtuple

from torchvision.models import vgg
import numpy as np
import main
import torch, torchvision
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import torchvision.models as models
import torch.nn as nn


class BigEncoder(nn.Module):
    def __init__(self, resolution=(28, 28), channels=1, compression_size=1):
        super(BigEncoder, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, (2, 2), stride=(2, 2))
        self.bn1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.conv2 = nn.Conv2d(self.conv1.out_channels, self.conv1.out_channels * 2, (2, 2), stride=(2, 2))
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3 = nn.Conv2d(self.conv2.out_channels, self.conv2.out_channels * 2, (2, 2), stride=(2, 2))
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)
        self.conv4 = nn.Conv2d(self.conv3.out_channels, self.conv3.out_channels, (2, 2))
        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)
        self.conv5 = nn.Conv2d(self.conv3.out_channels, self.conv3.out_channels, (2, 2))
        self.bn5 = nn.BatchNorm2d(self.conv4.out_channels)
        self.fc1 = nn.Linear(self.conv_width(resolution) ** 2 * self.conv3.out_channels,
                             compression_size)

        self.default_activation = nn.LeakyReLU()
        self.final_activation = nn.Sigmoid()

    def conv_width(self, resolution):
        h = resolution[0]
        for _ in range(3):
            h, _ = self.conv_out_shape(h, kernel_size=4, stride=2)
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

        img1d = self.conv4(img2c)
        img2d = self.default_activation(img1d)

        img1e = self.conv5(img2d)
        img2e = self.default_activation(img1e)

        vector = img2e.view(batch_size, -1)  # image 2 vec
        out = self.fc1(vector)
        out_normalizer = self.final_activation(out)

        return out_normalizer

    def conv_out_shape(self, h_w, kernel_size=3, stride=2, pad=0):
        h_w, kernel_size, stride, pad = self.int_to_tuple(h_w), \
                                        self.int_to_tuple(kernel_size), \
                                        self.int_to_tuple(stride), \
                                        self.int_to_tuple(pad)

        h = np.floor((h_w[0] + (2 * pad[0]) - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int)
        w = np.floor((h_w[1] + (2 * pad[1]) - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int)

        return h, w

    @staticmethod
    def int_to_tuple(x):
        return x if isinstance(x, tuple) else (x, x)


###########################################################################################################


writer = SummaryWriter()
encoder = BigEncoder(resolution=(28, 28), channels=1, compression_size=10)
device = 'cuda'
data_loader = main.data_loader
encoder.to(device)
enc_optim = torch.optim.Adam(encoder.parameters(), lr=.0001)

ENC_PATH = Path('Trained_GANS_4_INFERENCE_dont_touch/mnist_classifier_encoder_d_part')


###########################################################################################################

def trainer():
    epochs = 9
    idxer = iter(range(int(1e10)))
    for epoch in range(epochs):
        for idx, (images, labels) in enumerate(data_loader):
            index = idxer.__next__()
            images.to(device)
            pred = encoder(images)
            preds_soft = nn.Softmax(dim=-1)(pred)
            loss = nn.BCELoss()(preds_soft, hot(labels))
            loss.backward()
            enc_optim.step()
            enc_optim.zero_grad()
            writer.add_scalar('section_D_classifier_LOSS', loss.item(), index)
            if idx % 2_000 == 1_999:
                save_model()
            if idx % 5_00 == 0:
                accuracy = tester()
                writer.add_scalar('section_D_classifier_accuracy', accuracy, int(index / 500))


def hot(labels):
    hot_labels = torch.zeros(len(labels), 10)
    for idx, label in enumerate(labels):
        hot_labels[idx][label] = 1
    hot_labels_ = hot_labels.to(device)
    return hot_labels_


def tester():
    total = errors = 0
    for idx, (images, labels) in enumerate(data_loader):
        with torch.no_grad():
            pred_s = encoder(images)
            preds_soft = nn.Softmax(dim=-1)(pred_s)
            pred_labels = torch.argmax(preds_soft, dim=-1)
            errors += sum([1 for i in range(pred_labels.shape[0]) if pred_labels[i] != labels[i]])
            total += pred_labels.shape[0]
        if total >= 60:
            break
    print(errors / total)
    return errors / total


def save_model():
    torch.save(encoder.state_dict(), ENC_PATH)


if __name__ == '__main__':
    trainer()
