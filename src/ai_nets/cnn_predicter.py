import gzip
from os import listdir, makedirs, path

import numpy as np
from tqdm import tqdm

from src.ai_nets.pretrained_efficientnet import dataload, get_capped_model, preprocess
from src.helper.utils import PATH_TO_CACHE_FOLDER, PATH_TO_MODEL_FOLDER, PATH_TO_TEST_FOLDER, print_red


def predict(cnn, paths_to_predict_files: str, multiplier: int = 1):
    predict_folder = path.join(PATH_TO_CACHE_FOLDER, 'pre_predicted')
    if not path.isdir(predict_folder):
        makedirs(predict_folder)

    for file_path in tqdm(paths_to_predict_files, 'pre predicting progress'):

        if not path.isfile(file_path) or not file_path.endswith('.hdf5'):
            continue

        _, input_tensor, H1, L1 = dataload(file_path)
        tta = preprocess(multiplier, input_tensor, H1, L1)
        out = cnn(tta)

        file_name = path.basename(file_path)[:-5]
        with gzip.open(path.join(predict_folder, file_name) + '.npy.gz', 'wb') as gz_file:
            np.save(gz_file, out.cpu().detach().numpy())


if __name__ == '__main__':
    no_cw_folder = path.join(PATH_TO_TEST_FOLDER, 'no_cw_hdf5')
    cw_folder = path.join(PATH_TO_TEST_FOLDER, 'cw_hdf5')

    files = [path.join(no_cw_folder, file_name) for file_name in listdir(no_cw_folder)]
    files += [path.join(cw_folder, file_name) for file_name in listdir(cw_folder)]

    cnn = get_capped_model(path.join(PATH_TO_MODEL_FOLDER, 'model_best.pth'))
    predict(cnn, files)
