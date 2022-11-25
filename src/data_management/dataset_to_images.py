import os
from cProfile import label

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from src.data_management.dataset import GODataset
from src.helper.utils import PATH_TO_TEST_FOLDER, PATH_TO_TRAIN_FOLDER

dataset = GODataset(PATH_TO_TEST_FOLDER)
loader = DataLoader(dataset, batch_size=1)

# no_cw_path = f'{PATH_TO_TRAIN_FOLDER}/images/no_cw/'
# cw_path = f'{PATH_TO_TRAIN_FOLDER}/images/cw/'
no_cw_path = f'{PATH_TO_TRAIN_FOLDER}/images/no_cw/'
cw_path = f'{PATH_TO_TRAIN_FOLDER}/images/cw/'

if not os.path.isdir(f'{PATH_TO_TRAIN_FOLDER}/images'):
    os.makedirs(no_cw_path)
    os.makedirs(cw_path)

def normalize(img):
    max_value = np.max(img)
    min_value = np.min(img)
    return ((img - min_value) * 255 / (max_value - min_value)).astype('uint8')

for i, data in enumerate(loader, 0):
    inputs, labels = data

    image_data = np.array(inputs)[0]
    normalized_image_data = normalize(image_data)
    result = np.transpose(np.array(normalized_image_data), axes=(1, 2, 0))
    im = Image.fromarray(result)

    if labels[0] == 0:
        im.save(f'{no_cw_path}{i}.png')
    else:
        im.save(f'{cw_path}{i}.png')
