from datetime import datetime
from os import path

import timm
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import cv2

from src.ai_nets.trainer import GOTrainer
from src.data_management.datasets.realistic_dataset import GORealisticNoiseDataset
from src.helper.utils import PATH_TO_LOG_FOLDER, PATH_TO_MODEL_FOLDER, PATH_TO_SOURCE_FOLDER, get_df_dynamic_noise, get_df_signal, print_blue, print_green, print_red, print_yellow, normalize_image


epoch_index = 0
epoch_changed = True

class GOHadamardLayer(torch.nn.Module):
    def __init__(self, 
                 in_features, 
                 device, 
                 path_to_weight_image=path.join(PATH_TO_SOURCE_FOLDER, 'ai_nets', 'pretrained_weights', 'signals_pretrained.png')):
        super().__init__()

        weight_data = cv2.imread(path_to_weight_image, cv2.IMREAD_GRAYSCALE) / 255.0
        tensor = torch.tensor(weight_data, device=device, dtype=torch.float)
        self.weight = torch.nn.Parameter(tensor)
    
    def forward(self, x):
        return torch.mul(self.weight, x)


class GODenseMaxPoolModel(torch.nn.Module):
    def __init__(self, input_shape, batch_size, model, device):
        super().__init__()
        self.weighting = GOHadamardLayer(in_features=input_shape, device=device)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=7, stride=2, padding=0, dilation=3)
        # torch.nn.Linear(in_features=np.prod(input_shape), out_features=np.prod(input_shape), device=device)
        self.model = timm.create_model(model, num_classes=1, in_chans=2).to(device)

        self.batch_size = batch_size
        # self.input_shape = input_shape
        self.height = input_shape[-1]
        self.width = input_shape[-2]
    
    def forward(self, X): # shape (batch_size, detectors, height, width)
        global epoch_index, epoch_changed
        weighted = self.weighting(X)
        
        #linearized = pooled.reshape(self.batch_size, 2, -1)
        #weighted = self.weighting(linearized)
        #weighted = weighted.reshape(self.batch_size, 2, self.width, self.height)
        pooled = self.max_pool(weighted)
        
        if epoch_changed:
            epoch_changed = False
            cv2.imwrite(f'./gravitational_oceans/tmp/weights_{epoch_index}.png', normalize_image(self.weighting.weight.cpu().detach().numpy()))
            print_red('saved')
        
        return self.model(pooled)


class GOPlainCNNTrainer(GOTrainer):

    def __init__(self,
                 epochs: int = 15,
                 batch_size: int = 8,
                 lr: float = 0.0029125088766753125,
                 dropout: float = 0.1,
                 max_grad_norm: float = 13.644276196745786,
                 model: str = 'resnext50_32x4d',
                 logging: bool = True,
                 dataset_class = GORealisticNoiseDataset,
                 signal_strength: float = 1.0) -> None:

        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.dropout = dropout
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
        print_blue(self.model)
        #return GODenseMaxPoolModel((35, 199), self.batch_size, self.model, self.device)
        return timm.create_model(self.model, num_classes=1, in_chans=2, pretrained=True, drop_rate=0.1).to(self.device)
    
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
        global epoch_index, epoch_changed

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
            epoch_changed = True
            epoch_index = epoch
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
            print_blue('accuracy:', accuracy)

            if accuracy > max_accuracy:
                torch.save(model.state_dict(), f'{PATH_TO_MODEL_FOLDER}/plain_model.pth')
                max_accuracy = accuracy

            if self.logging:
                self.writer.add_scalar(f'val/loss', loss, epoch)
                self.writer.add_scalar(f'val/accuracy', accuracy, epoch)
                self.writer.add_scalar(f'val/max_accuracy', max_accuracy, epoch)

        return max_accuracy, accuracies


if __name__ == '__main__':
    #file_path = path.join(PATH_TO_CACHE_FOLDER, f'signal_strength_over_accuracy_with_dense_{datetime.now()}.csv')

    GOPlainCNNTrainer(logging=True, 
                    signal_strength=0.17, 
                    epochs=17, 
                    lr=0.000139, 
                    max_grad_norm=7.639,
                    model='inception_v4').train()

    # for f in np.linspace(0.21, 0.17, 5):
    #     max_accuracy, accuracies = GOPlainCNNTrainer(logging=False, signal_strength=f).train()
    #     with open(file_path, 'a') as file:
    #         file.write(f'{f},{max_accuracy},{",".join([str(x.numpy()) for x in accuracies])}\n')
