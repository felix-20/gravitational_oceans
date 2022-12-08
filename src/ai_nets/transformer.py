# https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1

import math
from datetime import datetime
from os import path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data_management.better_crnn_dataset import GOBetterCRNNDataset
from src.helper.utils import PATH_TO_LOG_FOLDER, PATH_TO_MODEL_FOLDER, print_blue, print_green, print_red, print_yellow

# parameters
sequence_length = 32
batch_size = 16
dim_model = 2048 # (closest power of two to shape of data)
epochs = 1

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pos_encoding',pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class GOTransformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/p/c80afbc9ffb1/
    """
    # Constructor
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

        # INFO
        self.model_type = 'Transformer'
        self.dim_model = dim_model

        # LAYERS
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.linear = nn.Linear(dim_model, num_tokens)

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        #src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embed_token(tgt)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1,0,2)
        tgt = tgt.permute(1,0,2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        linear_out = self.linear(transformer_out)

        return linear_out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)

    def embed_token(self, token):
        return self.embedding(token) * math.sqrt(self.dim_model)

def add_sequence_tokens(batch):
    if isinstance(batch[0], (torch.LongTensor, torch.cuda.LongTensor)):
        return torch.stack([torch.concat((SOS_TOKEN, item, EOS_TOKEN)) for item in batch])
    elif isinstance(batch[0], (torch.FloatTensor, torch.cuda.FloatTensor)):
        return torch.stack([torch.concat((SOS_TOKEN_EMBEDDED, item, EOS_TOKEN_EMBEDDED)) for item in batch])
    else:
        return batch

def features_to_embedding_vectors(features):
    # 192, 12, 115 -> 5, 52992
    split_and_flattened = torch.reshape(features, (sequence_length, -1))
    # 5, 52992 -> 5, 512
    embedded = split_and_flattened[:, :dim_model]
    return embedded * math.sqrt(dim_model)

def train_loop(model, opt, loss_fn, dataloader):
    model.train()
    total_loss = 0

    for x, y in dataloader:
        # convert from a multi-dimensional feature vector to a simple embedding-vector
        x = torch.stack([features_to_embedding_vectors(item) for item in x])

        x = x.to(device)
        y = y.type(torch.long).to(device)

        # prepend and append the sequence tokens
        x = add_sequence_tokens(x)
        y = add_sequence_tokens(y)

        # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
        y_input = y[:,:-1]
        y_expected = y[:,1:]

        # Get mask to mask out the next words
        sequence_length = y_input.size(1)
        tgt_mask = model.get_tgt_mask(sequence_length).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(x, y_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)
        loss = loss_fn(pred, y_expected)

        opt.zero_grad()
        loss.backward(retain_graph=True)
        opt.step()

        total_loss += loss.detach().item()

    return total_loss / len(dataloader)

def validation_loop(model, loss_fn, dataloader, epoch: int):
    model.eval()
    total_loss = 0
    total_accuracy_complete = 0
    total_accuracy_start = 0
    c_time = 0

    with torch.no_grad():
        for x, y in dataloader:
            # convert from a multi-dimensional feature vector to a simple embedding-vector
            x = torch.stack([features_to_embedding_vectors(item) for item in x])

            x = x.to(device)
            y = y.type(torch.long).to(device)

            # prepend and append the sequence tokens
            x = add_sequence_tokens(x)
            y = add_sequence_tokens(y)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:,:-1]
            y_expected = y[:,1:]

            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(device)

            # Standard training except we pass in y_input and src_mask
            pred = model(x, y_input, tgt_mask)

            # get accuracy
            _, max_index = torch.max(pred, dim=2)
            predicted = max_index.flatten()
            expected = y_expected.flatten()
            correct_complete = 0
            correct_start = 0
            sequence_length_dec = sequence_length - 1
            for i in range(sequence_length_dec):
                if predicted[i] == expected[i]:
                    correct_complete += 1
                    if correct_start == i:
                        correct_start += 1
            total_accuracy_complete += correct_complete / sequence_length_dec
            total_accuracy_start += correct_start / sequence_length_dec

            writer.add_scalar(f'total_acc/epoch_{epoch}', correct_complete / sequence_length_dec, c_time)
            writer.add_scalar(f'total_acc_start/epoch_{epoch}', correct_start / sequence_length_dec, c_time)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()

            c_time += 1

    total_loss /= len(dataloader)
    total_accuracy_complete /= len(dataloader)
    total_accuracy_start /= len(dataloader)

    return total_loss, total_accuracy_complete, total_accuracy_start

def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, writer):
    print_green('Training and validating model')
    max_accuracy_start = 0.0
    epoch_threshold = 20
    for epoch in tqdm(range(epochs), 'Epochs'):

        train_loss = train_loop(model, opt, loss_fn, train_dataloader)

        validation_loss, acc_complete, acc_start = validation_loop(model, loss_fn, val_dataloader, epoch)

        writer.add_scalar('loss/training', train_loss, epoch)
        writer.add_scalar('loss/validation', validation_loss, epoch)
        writer.add_scalar('accuracy/complete', acc_complete, epoch)
        writer.add_scalar('accuracy/start', acc_start, epoch)

        if epoch > epoch_threshold and acc_start > max_accuracy_start:
            torch.save(model, f'{PATH_TO_MODEL_FOLDER}/transformer_{epoch}_{acc_start}_{datetime.now().strftime("%Y-%m-%d_%H:%M")}.pt')
            max_accuracy_start = acc_start


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    dataset = GOBetterCRNNDataset(sequence_length=sequence_length)
    train_set, val_set = torch.utils.data.random_split(dataset, [round(len(dataset) * 0.8), round(len(dataset) * 0.2)])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f'Using device {device}')

    model_params = {
        'num_tokens': 4,
        'dim_model': dim_model,
        'num_heads': 2,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dropout_p': 0.1
    }

    model = GOTransformer(**model_params).to(device)

    SOS_TOKEN = torch.tensor([2]).to(device)
    EOS_TOKEN = torch.tensor([3]).to(device)
    SOS_TOKEN_EMBEDDED = model.embed_token(SOS_TOKEN)
    EOS_TOKEN_EMBEDDED = model.embed_token(EOS_TOKEN)

    model = GOTransformer(**model_params).to(device)

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    writer = SummaryWriter(path.join(PATH_TO_LOG_FOLDER, 'runs', f'transformer_{str(datetime.now())}'))
    writer.add_text('hyperparameters/batch_size', str(batch_size))
    writer.add_text('hyperparameters/sequence_length', str(sequence_length))
    writer.add_text('hyperparameters/dim_model', str(dim_model))
    writer.add_text('hyperparameters/epochs', str(epochs))

    time_before = datetime.now()
    fit(model, opt, loss_fn, train_loader, val_loader, epochs, writer)
    time_after = datetime.now()
    time_difference = time_after - time_before
    print(str(time_difference))
    writer.add_text('metrics/training_time', str(time_difference))
