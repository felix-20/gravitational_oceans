# https://github.com/dredwardhyde/crnn-ctc-loss-pytorch
import sys
from datetime import datetime
from itertools import groupby

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter
from colorama import Fore
from torchvision import datasets, transforms
from tqdm import tqdm
from os import path

from src.data_management.crnn_dataset import GOCRNNDataset
from src.helper.utils import PATH_TO_MODEL_FOLDER, PATH_TO_LOG_FOLDER, print_blue, print_green, print_red, print_yellow

epochs = 5
num_classes = 3
blank_label = 2
image_height = 360
gru_hidden_size = 128
gru_num_layers = 2
cnn_output_height = 21
cnn_output_width = 43
digits_per_sequence = 2
number_of_sequences = 2566
dataset_sequences = []
dataset_labels = []
writer = SummaryWriter(path.join(PATH_TO_LOG_FOLDER, 'runs', str(datetime.now())))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} for training')

seq_dataset = GOCRNNDataset(sequence_length=digits_per_sequence)# data_utils.TensorDataset(dataset_data, dataset_labels)
train_set, val_set = torch.utils.data.random_split(seq_dataset,
                                                   [round(len(seq_dataset) * 0.8), round(len(seq_dataset) * 0.2)])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)

# ================================================= MODEL ==============================================================
class CRNN(nn.Module):

    def __init__(self):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.norm1 = nn.InstanceNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=2)
        self.norm2 = nn.InstanceNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.norm3 = nn.InstanceNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=2)
        self.norm4 = nn.InstanceNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2)
        self.norm5 = nn.InstanceNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=2)
        self.norm6 = nn.InstanceNorm2d(128)
        self.gru_input_size = cnn_output_height * 128
        self.gru = nn.GRU(self.gru_input_size, gru_hidden_size, gru_num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(gru_hidden_size * 2, num_classes)

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
        out = out.reshape(batch_size, -1, self.gru_input_size)
        out, _ = self.gru(out)
        out = torch.stack([F.log_softmax(self.fc(out[i]), dim=-1) for i in range(out.shape[0])])
        return out


model = CRNN().to(device)
criterion = nn.CTCLoss(blank=blank_label, reduction='mean', zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ================================================ TRAINING MODEL ======================================================
for num_epoch in range(epochs):
    # ============================================ TRAINING ============================================================
    train_correct = 0
    train_total = 0

    time = 0
    for x_train, y_train in tqdm(train_loader,
                                 position=0, leave=True,
                                 file=sys.stdout, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.GREEN, Fore.RESET)):
        batch_size = x_train.shape[0]  # x_train.shape == torch.Size([64, 28, 140])
        x_train = x_train.view(x_train.shape)
        optimizer.zero_grad()
        y_pred = model(x_train.to(device).float())
        y_pred = y_pred.permute(1, 0, 2)  # y_pred.shape == torch.Size([64, 32, 11])
        input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
        target_lengths = torch.IntTensor([len(t) for t in y_train])

        loss = criterion(y_pred, y_train, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()
        _, max_index = torch.max(y_pred, dim=2)  # max_index.shape == torch.Size([32, 64])

        for i in range(batch_size):
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())  # len(raw_prediction) == 32
            prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])

            writer.add_scalar(f'predicition_length/epoch_{num_epoch}', len(prediction), time)
            
            partial_correct = 0
            partial_total = min(len(prediction), len(y_train[i]))
            if partial_total != 0:
                for j  in range(partial_total):
                    if prediction[j] == y_train[i][j]:
                        partial_correct += 1
                
                writer.add_scalar(f'partial_correct/epoch_{num_epoch}', partial_correct / partial_total, time)

            if len(prediction) == len(y_train[i]) and torch.all(prediction.eq(y_train[i])):
                train_correct += 1
            train_total += 1

            time += 1
        
                
    print('TRAINING. Correct: ', train_correct, '/', train_total, '=', train_correct / train_total)

    # ============================================ VALIDATION ==========================================================
    val_correct = 0
    val_total = 0
    for x_val, y_val in tqdm(val_loader,
                             position=0, leave=True,
                             file=sys.stdout, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET)):
        batch_size = x_val.shape[0]
        x_val = x_val.view(x_val.shape)
        y_pred = model(x_val.to(device).float())
        y_pred = y_pred.permute(1, 0, 2)
        input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
        target_lengths = torch.IntTensor([len(t) for t in y_val])
        criterion(y_pred, y_val, input_lengths, target_lengths)
        _, max_index = torch.max(y_pred, dim=2)
        for i in range(batch_size):
            raw_prediction = list(max_index[:, i].detach().cpu().numpy())
            prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])
            if len(prediction) == len(y_val[i]) and torch.all(prediction.eq(y_val[i])):
                val_correct += 1
            val_total += 1
    print('TESTING. Correct: ', val_correct, '/', val_total, '=', val_correct / val_total)

# ============================================ TESTING =================================================================
number_of_test_imgs = 10
test_loader = torch.utils.data.DataLoader(val_set, batch_size=number_of_test_imgs, shuffle=True)
test_preds = []
(x_test, y_test) = next(iter(test_loader))
y_pred = model(x_test.view(x_test.shape).to(device).float())
y_pred = y_pred.permute(1, 0, 2)
_, max_index = torch.max(y_pred, dim=2)
for i in range(x_test.shape[0]):
    raw_prediction = list(max_index[:, i].detach().cpu().numpy())
    prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])
    test_preds.append(prediction)

torch.save(model, f'{PATH_TO_MODEL_FOLDER}/crnn_{datetime.now().strftime("%Y-%m-%d_%H:%M")}.pt')

# for j in range(len(x_test)):
#     mpl.rcParams["font.size"] = 8
#     plt.imshow(x_test[j], cmap='gray')
#     mpl.rcParams["font.size"] = 18
#     plt.gcf().text(x=0.1, y=0.1, s="Actual: " + str(y_test[j].numpy()))
#     plt.gcf().text(x=0.1, y=0.2, s="Predicted: " + str(test_preds[j].numpy()))
#     plt.show()
#     plt.savefig('')
