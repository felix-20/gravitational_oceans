import torch
import torch.nn as nn

from os import path
import math
from datetime import datetime

from src.helper.utils import print_blue, print_green, print_red, print_yellow, PATH_TO_MODEL_FOLDER, PATH_TO_TRAIN_FOLDER, PATH_TO_TEST_FOLDER
from src.data_management.better_crnn_dataset import GOBetterCRNNDataset
import src.ai_nets.cnn_predicter as GOCNNPredictor

ALL_FILES = path.listdir(PATH_TO_TRAIN_FOLDER)
ALL_FILES += path.listdir(PATH_TO_TEST_FOLDER)

GOCNNPredictor.predict(ALL_FILES)


