# https://github.com/sunnynevarekar/pytorch-saliency-maps/blob/master/Saliency_maps_in_pytorch.ipynb

from copy import deepcopy
from os.path import join

import matplotlib.pyplot as plt
import timm
import torch
import numpy as np

from src.data_management.datasets.realistic_dataset import GORealisticNoiseDataset
from src.helper.dynamic_noise_visualisation import get_one_sample
from src.helper.utils import PATH_TO_CACHE_FOLDER, PATH_TO_MODEL_FOLDER, print_green, print_yellow


def saliency(img, model):
    # we don't need gradients w.r.t. weights for a trained model
    for param in model.parameters():
        param.requires_grad = False

    # set model in eval mode
    model.eval()

    # keep the original for displaying it later
    org = deepcopy(img)[0]

    # bring img to right shape
    model_input = img
    model_input.unsqueeze_(0)

    # we want to calculate gradient of higest score w.r.t. input
    # so set requires_grad to True for input
    model_input.requires_grad = True

    # forward pass to calculate predictions
    preds = model(model_input)
    score, indices = torch.max(preds, 1)

    # backward pass to get gradients of score predicted class w.r.t. input image
    score.backward()

    # get max along channel axis
    slc, _ = torch.max(torch.abs(model_input.grad[0]), dim=0)

    # normalize to [0..1]
    slc = (slc - slc.min())/(slc.max()-slc.min())

    # apply inverse transform on image
    # plot image and its saliency map
    plt.figure(figsize=(40, 20))
    plt.subplot(1, 3, 1)
    plt.imshow(org.cpu(), cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 3, 2)
    plt.imshow(slc.cpu(), cmap=plt.cm.hot, interpolation='none')
    plt.xticks([])
    plt.yticks([])

    image_tensor = torch.cat([org.cpu().unsqueeze(0), slc.cpu().unsqueeze(0), slc.cpu().unsqueeze(0)], dim=0)
    image_array = np.transpose(np.clip(image_tensor.numpy(), 0, 1), (1, 2, 0))
    plt.subplot(1, 3, 3)
    plt.imshow(image_array, interpolation='none')
    plt.xticks([])
    plt.yticks([])

    plt.savefig(join(PATH_TO_CACHE_FOLDER, 'sali.png'))


if __name__ == '__main__':
    model_path = join(PATH_TO_MODEL_FOLDER, 'plain_model.pth')
    model_type = 'inception_v4'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = timm.create_model(model_type, pretrained=False, num_classes=1, in_chans=2).to(device)
    model.load_state_dict(torch.load(model_path))

    image = torch.tensor(get_one_sample(should_normalize=False)).to(device)
    model = model.to(device)

    saliency(image, model)
