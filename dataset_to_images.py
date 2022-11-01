from cProfile import label
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os

from dataset import GODataset

dataset = GODataset('./data')
loader = DataLoader(dataset, batch_size=1)

no_cw_path = './data/images/no_cw/'
cw_path = './data/images/cw/'

if not os.path.isdir('./data/images'):
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
