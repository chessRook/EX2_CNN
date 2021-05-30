from builtins import int

import section_b_inverse_gan
import main
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transform
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

trained_generator, disc_ = section_b_inverse_gan.model_loader()
inverser = section_b_inverse_gan.inverser_loader()
data_loader = main.data_loader

loss = nn.MSELoss()


def freezer(model):
    for param in model.parameters():
        param.requires_grad = False


def z_searcher(img_masked, img):
    img_masked.requires_grad = False
    img_masked = img_masked.to('cuda')
    z = inverser(img_masked)
    z = z.detach()
    z_copy = torch.clone(z).to('cuda')
    optimizer = torch.optim.Adam((z_copy,), lr=.001)
    trained_generator.train(False)
    for idx in tqdm(range(int(1e3))):
        pred = trained_generator(z_copy)
        loss_ten = loss(img_masked, pred)
        loss_ten.backward()
        optimizer.step()
        optimizer.zero_grad()
        main.writer.add_scalar('section_C_LOSS', loss_ten.item(), idx)
    restored_img = trained_generator(z_copy)
    restored_pil_img = transform.ToPILImage()(restored_img[0])
    results = {'restored': restored_pil_img, 'img': img, 'masked_img': img_masked}
    return results, z_copy


def _main_():
    mask_methods = {'random_mask': random_mask, 'center_mask': center_mask, 'large_center_mask': large_center_mask,
                    'small_center_mask': small_center_mask}
    freezer(inverser)
    for mask_name, mask_method in mask_methods.items():
        restore_with_me(mask_name, mask_method, num_of_images=4)


def restore_with_me(mask_name, mask_method, num_of_images=4):
    results_by_time = {}
    for idx, (img, label) in zip(range(num_of_images), data_loader):
        img.to('cuda')
        img_masked = mask_method(img)
        results_dict, __ = z_searcher(img_masked, img)
        results_by_time[idx] = results_dict
    display(results_by_time, mask_name)


def display(results_by_time, mask_name):
    fig = plt.figure()
    plt.title(f'{mask_name}'.capitalize(), loc='left', y=-.08)
    plt.axis('off')
    total_time = max(results_by_time.keys()) + 1
    for time in results_by_time:

        present_results = results_by_time[time]

        ten1 = present_results['img'][0].squeeze()
        ax1 = fig.add_subplot(total_time, 3, 1 + 3 * time)
        plt.axis('off')
        ax1.imshow(ten1)

        ten2 = present_results['masked_img'][0].squeeze()
        ax2 = fig.add_subplot(total_time, 3, 2 + 3 * time)
        plt.axis('off')
        ax2.imshow(ten2.cpu())

        ten3 = transform.ToTensor()(present_results['restored'])[0].squeeze()
        ax3 = fig.add_subplot(total_time, 3, 3 + 3 * time)
        plt.axis('off')
        ax3.imshow(ten3)

        if time == 0:
            ax1.title.set_text('img')
            ax2.title.set_text('masked_img')
            ax3.title.set_text('restored')

    plt.show()


def pil_2_plt(pil_img):
    np_array = np.asarray(pil_img)
    img_tensor = torch.tensor(np_array)
    return img_tensor


def small_center_mask(image):
    out_image = image.clone()
    i_center, j_center = int(image.shape[-1] / 2), int(image.shape[-1] / 2)
    i_window, j_window = int(image.shape[-1] / 8), int(image.shape[-1] / 8)
    mask = torch.zeros(size=(i_window * 2, j_window * 2))
    out_image[:, 0, i_center - i_window: i_center + i_window, j_center - j_window: j_center + j_window] = mask
    return out_image


def large_center_mask(image):
    out_image = image.clone()
    i_center, j_center = int(image.shape[-1] / 2), int(image.shape[-1] / 2)
    i_window, j_window = int(image.shape[-1] / 4), int(image.shape[-1] / 4)
    mask = torch.zeros(size=(i_window * 2, j_window * 2))
    out_image[:, 0, i_center - i_window: i_center + i_window, j_center - j_window: j_center + j_window] = mask
    return out_image


def center_mask(image):
    out_image = image.clone()
    i_center, j_center = int(image.shape[-1] * .34), int(image.shape[-1] * .34)
    i_window, j_window = int(image.shape[-1] * .43), int(image.shape[-1] * .43)
    for i in range(i_window):
        for j in range(j_window):
            for k in range(main.batch_size):
                out_image[k, 0, i_center + i, j_center + j] = random.random()
    return out_image


def random_mask(image):
    out_image = image.clone()
    for t in range(28 * 28 * 2):
        img_num = random.randint(0, 4)
        i = random.randint(0, 27)
        j = random.randint(0, 27)
        out_image[img_num, 0, i, j] = 0
    return out_image


if __name__ == '__main__':
    _main_()
