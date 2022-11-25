import torch
import torch.nn as nn
import torch.nn.functional as F

from pretrained_efficientnet import get_capped_model

class CRNN(nn.Module):

    def __init__(self,
                 cnn : nn.Module,
                 sequence_length : int = 5,
                 cnn_output_height : int = 0,
                 gru_hidden_size : int = 0,
                 gru_num_layers : int = 0,
                 num_classes : int = 0):
        super(CRNN, self).__init__()

        self.cnn = cnn
        self.sequence_length = sequence_length

        self.gru_input_size = cnn_output_height * 64
        self.gru = nn.GRU(10036, gru_hidden_size, gru_num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(gru_hidden_size * 2, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]

        out = self.cnn.forward(x)
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