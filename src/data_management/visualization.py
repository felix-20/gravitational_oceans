import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import cm
from PIL import Image

from src.helper.utils import open_hdf5_file, print_blue, print_green, print_red, print_yellow


def get_gaps(timestamp, comparing: bool = True):
    gaps = []
    for i in range(len(timestamp)-1):
        gap_size = timestamp[i+1] - timestamp[i]
        if comparing or gap_size != 1800:
            gaps += [gap_size]
    return gaps


def draw_gap_sizes_color(data, time, file_id):
    real = np.zeros(data['amplitudes'].shape)

    print_red(real.shape)
    for i in range(len(time) - 2, -1, -1):
        gap_size = time[i+1] - time[i]
        real = np.c_[real[:,:i], np.full(real.shape[0], gap_size), real[:,i:]]

    max_value = np.max(real)
    min_value = np.min(real)
    real = (real - min_value) / (max_value - min_value)

    im = Image.fromarray(np.uint8(cm.gist_gray(real)*255))
    im.save(f'tmp/{file_id}_gaps.png')


def draw_gap_size_histogram(time, file_id):
    gaps = []
    for i in range(len(time)-1):
        gap_size = time[i+1] - time[i]
        gaps += [gap_size]

    print(gaps)

    plt.xlim([min(gaps) - 5, max(gaps) + 5])

    plt.hist(gaps, bins=200)
    plt.savefig(f'tmp/{file_id}_hist.png')


if __name__ == '__main__':
    # draw_gap_sizes_color()
    file_id = '42a3b2de1' #'518949f96'
    file_path = f'tmp/{file_id}.hdf5'

    data = open_hdf5_file(file_path)['h1']

    time = data['timestamps']
    total_time = time[-1] - time[0]

    print_green(total_time)
    print_yellow(len(time))

    draw_gap_size_histogram(time, file_id)
