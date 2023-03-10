from datetime import datetime
from os import path

import timm
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.ai_nets.trainer import GOTrainer
from src.data_management.datasets.realistic_dataset import GORealisticNoiseDataset
from src.helper.utils import PATH_TO_LOG_FOLDER, PATH_TO_MODEL_FOLDER, PATH_TO_CACHE_FOLDER, get_df_dynamic_noise, get_df_signal, print_blue, print_green, print_red, print_yellow


class GOPlainCNNTrainer(GOTrainer):

    def __init__(self,
                 epochs: int = 20,
                 batch_size: int = 8,
                 lr: float = 0.01,
                 max_grad_norm: float = 1.0,
                 model: str = 'resnet50',
                 logging: bool = True,
                 dataset_class = GORealisticNoiseDataset,
                 signal_strength: float = 1.0) -> None:

        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.model = model
        self.logging = logging
        self.dataset_class = dataset_class
        self.signal_strength = signal_strength
        self.input_shape = (360, 2000)

        if logging:
            self.writer = SummaryWriter(path.join(PATH_TO_LOG_FOLDER, 'runs', f'plain_cnn_{str(datetime.now())}'))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f'Training on {self.device}')

    def get_model(self):
        input_size = self.input_shape[1]# self.input_shape[0] * self.input_shape[1]
        dense = torch.nn.Linear(in_features=input_size, out_features=input_size, device=self.device)
        max_pool = torch.nn.MaxPool2d(kernel_size=7, stride=2, padding=0, dilation=3)
        model = timm.create_model(self.model, num_classes=1, in_chans=2).to(self.device)
        return torch.nn.Sequential(dense, max_pool, model)
    
    """   
    class FullModel():
        def __init__():
            self.dense = torch.nn.Linear(in_features=input_size, out_features=input_size, device=self.device)
            self.max_pool = torch.nn.MaxPool2d(kernel_size=7, stride=2, padding=0, dilation=3)
            self.model = timm.create_model(self.model, num_classes=1, in_chans=2).to(self.device)
        
        def call(self, input): # shape (batch_size, detectors, height, width)
            linearized = input.reshape(batch_size, 2, width * height)
            weighted = dense(linearized)
            2d_         
    """

    @torch.no_grad()
    def evaluate(self, model, dl_eval):
        model.eval()

        predictions = []
        labels = []
        signal_strengths = []

        for X, y, signal_strength in tqdm(dl_eval, desc='Eval', colour='#fc8403'):
            labels += [y]
            signal_strengths += [signal_strength]
            predictions += [model(X.to(self.device)).cpu().squeeze()]

        labels = torch.concat(labels)
        predictions = torch.concat(predictions)
        
        loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions, labels.float(), reduction='none').median().item()

        integer_predictions = torch.round(torch.sigmoid(predictions))
        correct = torch.sum(torch.eq(labels.bool(), integer_predictions.bool()))

        return (correct / len(labels), loss, predictions, labels, signal_strengths)

    def train(self):
        noise_files = get_df_dynamic_noise()
        signal_files = get_df_signal()

        np.random.shuffle(noise_files)
        np.random.shuffle(signal_files)

        noise_files_train = noise_files[:int(len(noise_files)*0.8)]
        noise_files_eval = noise_files[int(len(noise_files)*0.8):]

        signal_files_train = signal_files[:int(len(signal_files)*0.8)]
        signal_files_eval = signal_files[int(len(signal_files)*0.8):]

        dataset_train = self.dataset_class(
            len(signal_files_train),
            noise_files_train,
            signal_files_train,
            signal_strength=self.signal_strength
        )

        dataset_eval = self.dataset_class(
            len(signal_files_eval),
            noise_files_eval,
            signal_files_eval,
            signal_strength=self.signal_strength
        )

        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, drop_last=True)
        dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=self.batch_size, drop_last=True)

        model = self.get_model()
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)

        max_accuracy = 0
        accuracies = []
        for epoch in range(self.epochs):
            print_green(f'Training Epoch {epoch}')
            for step, (X, y, signal_strength) in enumerate(tqdm(dataloader_train, desc='Train', colour='#6ea62e')):
                predictions = model(X.to(self.device)).squeeze()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(predictions, y.float().to(self.device))

                optim.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optim.step()

                if self.logging:
                    self.writer.add_scalar(f'loss/epoch_{epoch}', loss.item(), step)
                    self.writer.add_scalar(f'grad_norm/epoch_{epoch}', norm, step)
                    self.writer.add_scalar(f'logit/epoch_{epoch}', predictions.mean().item(), step)

            accuracy, loss = self.evaluate(model, dataloader_eval)[:2]
            accuracies += [accuracy]

            if accuracy > max_accuracy:
                torch.save(model.state_dict(), f'{PATH_TO_MODEL_FOLDER}/plain_model.pth')
                max_accuracy = accuracy

            if self.logging:
                self.writer.add_scalar(f'val/loss', loss, epoch)
                self.writer.add_scalar(f'val/accuracy', accuracy, epoch)
                self.writer.add_scalar(f'val/max_accuracy', max_accuracy, epoch)

        return max_accuracy, accuracies


if __name__ == '__main__':
    file_path = path.join(PATH_TO_CACHE_FOLDER, 'signal_strength_over_accuracy.csv')

    for f in np.linspace(0.21, 0.17, 5):
        max_accuracy, accuracies = GOPlainCNNTrainer(logging=False, signal_strength=f).train()
        with open(file_path, 'a') as file:
            file.write(f'{f},{max_accuracy},{",".join([str(x.numpy()) for x in accuracies])}\n')
