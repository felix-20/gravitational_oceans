import os

import cv2
import numpy as np
from tqdm import tqdm

from src.helper.utils import PATH_TO_SIGNAL_FOLDER, normalize_image, print_red


def combine_signals():
    all_signals = [os.path.join(PATH_TO_SIGNAL_FOLDER, f) for f in os.listdir(PATH_TO_SIGNAL_FOLDER)]

    result = np.zeros((360, 2000))
    for f in tqdm(all_signals):
        data = cv2.imread(f, cv2.IMREAD_GRAYSCALE)[:,:2000]

        result += data

    cv2.imwrite(os.path.join('.', 'gravitational_oceans', 'tmp', 'combined_signals.png'), normalize_image(result))


if __name__ == '__main__':
    combine_signals()
