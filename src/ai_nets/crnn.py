# https://github.com/dredwardhyde/crnn-ctc-loss-pytorch
import sys
from datetime import datetime
from itertools import groupby
from os import path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.transforms.functional as TF
from colorama import Fore
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm

from src.data_management.crnn_dataset import GOCRNNDataset
from src.helper.utils import PATH_TO_LOG_FOLDER, PATH_TO_MODEL_FOLDER, print_blue, print_green, print_red, print_yellow


class GOCRNNParameters:
    def __init__(self,
        gru_hidden_size: int = 128,
        epochs: int = 2,
        num_classes: int = 2,
        blank_label: int = 2,
        image_height: int = 360,
        gru_num_layers: int = 2,
        cnn_output_height: int = 21,
        cnn_output_width: int = 5,
        sequence_length: int = 5,
        number_of_sequences: int = 2566,
        batch_size: int = 8) -> None:
        
        self.gru_hidden_size = gru_hidden_size
        self.epochs = epochs
        self.num_classes = num_classes
        self.blank_label = blank_label
        self.image_height = image_height
        self.gru_num_layers = gru_num_layers
        self.cnn_output_height = cnn_output_height
        self.cnn_output_width = cnn_output_width
        self.sequence_length = sequence_length
        self.number_of_sequences = number_of_sequences
        self.batch_size = batch_size


# ================================================= MODEL ==============================================================
class CRNN(nn.Module):

    def __init__(self, params: GOCRNNParameters):
        super(CRNN, self).__init__()
        self.sequence_length = params.sequence_length
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(31, 31))
        self.norm1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(31, 31), stride=2)
        self.norm2 = nn.InstanceNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(31, 31))
        self.norm3 = nn.InstanceNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(31, 31), stride=2)
        self.norm4 = nn.InstanceNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=(6, 6), stride=2)
        self.norm5 = nn.InstanceNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=(5, 5), stride=2)
        self.norm6 = nn.InstanceNorm2d(64)
        self.gru_input_size = params.cnn_output_height * 64
        self.gru = nn.GRU(10036, params.gru_hidden_size, params.gru_num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(params.gru_hidden_size * 2, params.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.leaky_relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.leaky_relu(out)
        out = self.conv3(out)
        out = self.norm3(out)
        out = F.leaky_relu(out)
        out = self.conv4(out)
        out = self.norm4(out)
        out = F.leaky_relu(out)
        out = self.conv5(out)
        out = self.norm5(out)
        out = F.leaky_relu(out)
        out = self.conv6(out)
        out = self.norm6(out)
        out = F.leaky_relu(out)

        out = out.permute(0, 3, 2, 1)

        data_amount = out.shape[1] * out.shape[2] * out.shape[3]
        pad_length = self.sequence_length - (data_amount % self.sequence_length)
        data_amount_per_sequence = (data_amount + pad_length) // self.sequence_length

        out = out.reshape(batch_size, -1)
        out = F.pad(out, (0, pad_length), 'constant', 0)
        out = out.reshape(batch_size, self.sequence_length, data_amount_per_sequence)

        out, _ = self.gru(out)
        out = torch.stack([F.log_softmax(self.fc(out[i]), dim=-1) for i in range(out.shape[0])])

        return out


class GOCRNNTrainer:
    def __init__(self, params: GOCRNNParameters, model: nn.Module, dataset) -> None:
        self.epochs = params.epochs
        self.cnn_output_width = params.cnn_output_width
        self.blank_label = params.blank_label
        self.dataset_sequences = []
        self.dataset_labels = []
        self.writer = SummaryWriter(path.join(PATH_TO_LOG_FOLDER, 'runs', str(datetime.now())))

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {self.device} for training')

        train_set, self.val_set = torch.utils.data.random_split(dataset,
                                                        [round(len(dataset) * 0.8), round(len(dataset) * 0.2)])

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=params.batch_size, shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=1, shuffle=True)

        self.model = model.to(self.device)
        self.criterion = nn.CTCLoss(reduction='mean', zero_infinity=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self) -> None:
        # ================================================ TRAINING MODEL ======================================================
        for num_epoch in range(self.epochs):
            # ============================================ TRAINING ============================================================
            train_correct = 0
            train_total = 0

            time = 0
            for x_train, y_train in tqdm(self.train_loader,
                                        position=0, leave=True,
                                        file=sys.stdout, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.GREEN, Fore.RESET)):
                batch_size = x_train.shape[0]
                x_train = x_train.view(x_train.shape)
                self.optimizer.zero_grad()
                y_pred = self.model(x_train.to(self.device).float())
                y_pred = y_pred.permute(1, 0, 2)
                input_lengths = torch.IntTensor(batch_size).fill_(self.cnn_output_width)
                target_lengths = torch.IntTensor([len(t) for t in y_train])

                loss = self.criterion(y_pred, y_train, input_lengths, target_lengths)
                loss.backward()
                self.optimizer.step()

                _, max_index = torch.max(y_pred, dim=2)

                for i in range(batch_size):
                    prediction = torch.IntTensor(max_index[:, i].detach().cpu().numpy())

                    partial_correct = 0
                    partial_total = min(len(prediction), len(y_train[i]))
                    if partial_total != 0:
                        for j  in range(partial_total):
                            if prediction[j] == y_train[i][j]:
                                partial_correct += 1

                        self.writer.add_scalar(f'partial_correct/epoch_{num_epoch}', partial_correct / partial_total, time)

                    if len(prediction) == len(y_train[i]) and torch.all(prediction.eq(y_train[i])):
                        train_correct += 1
                    train_total += 1

                    time += 1


            print('TRAINING. Correct: ', train_correct, '/', train_total, '=', train_correct / train_total)

            # ============================================ VALIDATION ==========================================================
            val_correct = 0
            val_total = 0
            for x_val, y_val in tqdm(self.val_loader,
                                    position=0, leave=True,
                                    file=sys.stdout, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET)):
                batch_size = x_val.shape[0]
                x_val = x_val.view(x_val.shape)
                y_pred = self.model(x_val.to(self.device).float())
                y_pred = y_pred.permute(1, 0, 2)
                input_lengths = torch.IntTensor(batch_size).fill_(self.cnn_output_width)
                target_lengths = torch.IntTensor([len(t) for t in y_val])
                self.criterion(y_pred, y_val, input_lengths, target_lengths)
                _, max_index = torch.max(y_pred, dim=2)
                for i in range(batch_size):
                    prediction = torch.IntTensor(max_index[:, i].detach().cpu().numpy())

                    if len(prediction) == len(y_val[i]) and torch.all(prediction.eq(y_val[i])):
                        val_correct += 1
                    val_total += 1
            print('TESTING. Correct: ', val_correct, '/', val_total, '=', val_correct / val_total)

        # ============================================ TESTING =================================================================
        number_of_test_imgs = 10
        test_loader = torch.utils.data.DataLoader(self.val_set, batch_size=number_of_test_imgs, shuffle=True)
        test_preds = []
        (x_test, y_test) = next(iter(test_loader))
        y_pred = self.model(x_test.view(x_test.shape).to(self.device).float())
        y_pred = y_pred.permute(1, 0, 2)
        _, max_index = torch.max(y_pred, dim=2)
        for i in range(x_test.shape[0]):
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())
            prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != self.blank_label])
            test_preds.append(prediction)

        torch.save(self.model, f'{PATH_TO_MODEL_FOLDER}/crnn_{datetime.now().strftime("%Y-%m-%d_%H:%M")}.pt')

if __name__ == '__main__':
    params = GOCRNNParameters()
    dataset = GOCRNNDataset(sequence_length=params.sequence_length)
    GOCRNNTrainer(params, CRNN(params), dataset).train()
