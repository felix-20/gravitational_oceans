from os import getcwd, listdir, makedirs
from os.path import basename, join
from uuid import uuid4

import cv2

from src.helper.utils import PATH_TO_DYNAMIC_NOISE_FOLDER, PATH_TO_SIGNAL_FOLDER

# (360, 256, 3)


PATH_TO_NEW_NOISE = join(getcwd(), '2', 'noise', 'dynamic')

all_files = [join(PATH_TO_DYNAMIC_NOISE_FOLDER, f) for f in listdir(PATH_TO_DYNAMIC_NOISE_FOLDER)]

for f in all_files:
    img = cv2.imread(f)
    h1 = img[:, :256]
    l1 = img[:, 256:512]

    path = join(PATH_TO_NEW_NOISE, uuid4())
    makedirs(path)
    cv2.imwrite(join(path, 'H1.png'), h1)
    cv2.imwrite(join(path, 'L1.png'), l1)

PATH_TO_NEW_SIGNAL = join(getcwd(), '2', 'signal')

all_files = [join(PATH_TO_SIGNAL_FOLDER, f) for f in listdir(PATH_TO_SIGNAL_FOLDER)]

for f in all_files:
    img = cv2.imread(f)
    h1 = img[:, :256]
    cv2.imwrite(join(PATH_TO_NEW_SIGNAL, basename(f)), h1)
