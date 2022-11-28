import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ai_nets.crnn import GOCRNNParameters, GOCRNNTrainer
from src.ai_nets.pretrained_efficientnet import get_capped_model
from src.data_management.better_crnn_dataset import GOBetterCRNNDataset
from src.helper.utils import PATH_TO_MODEL_FOLDER, print_blue, print_green, print_red, print_yellow


class GOCRNN(nn.Module):

    def __init__(self,
                 cnn : nn.Module,
                 params : GOCRNNParameters):
        super(GOCRNN, self).__init__()

        self.cnn = cnn
        self.sequence_length = params.sequence_length

        self.gru_input_size = params.cnn_output_height * 64
        self.gru = nn.GRU(70657, params.gru_hidden_size, params.gru_num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(params.gru_hidden_size * 2, params.num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        out = self.cnn(x)

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


if __name__ == '__main__':
    params = GOCRNNParameters(batch_size = 64,
                              epochs=8)

    print_green('setting up crnn')
    cnn = get_capped_model(os.path.join(PATH_TO_MODEL_FOLDER, 'model_best.pth'))
    crnn = GOCRNN(cnn, params)

    print_green('prepearing training')
    dataset = GOBetterCRNNDataset(sequence_length=params.sequence_length)
    crnn_trainer = GOCRNNTrainer(params, crnn, dataset)

    print_green('execute trainer')
    crnn_trainer.train()
