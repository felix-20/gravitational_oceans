from datetime import datetime
from os import path

import json
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
from src.helper.utils import PATH_TO_LOG_FOLDER, PATH_TO_MODEL_FOLDER, PATH_TO_CACHE_FOLDER, get_df_dynamic_noise, get_df_signal, print_blue, print_green, print_red, print_yellow, normalize_image
from src.helper.crange import crange


class GOPlainCNNTrainer(GOTrainer):

    def __init__(self,
                 epochs: int = 17,
                 batch_size: int = 8,
                 lr: float = 0.000139,
                 dropout: float = 0.1,
                 max_grad_norm: float = 7.639,
                 model: str = 'inception_v4',
                 logging: bool = True,
                 dataset_class = GORealisticNoiseDataset,
                 signal_strength_upper: float = 0.17,
                 signal_strength_lower: float = 0.02) -> None:

        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.dropout = dropout
        self.model = model
        self.logging = logging
        self.dataset_class = dataset_class
        self.signal_strength_upper = signal_strength_upper
        self.signal_strength_lower = signal_strength_lower
        assert self.signal_strength_lower <= self.signal_strength_upper, \
            f'Signal strength upper needs to be higher or equal to signal strength lower, upper is {self.signal_strength_upper}, lower is {self.signal_strength_lower}'

        if logging:
            self.writer = SummaryWriter(path.join(PATH_TO_LOG_FOLDER, 'runs', f'plain_cnn_{str(datetime.now())}'))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f'Training on {self.device}')

    def get_model(self):
        # return GODenseMaxPoolModel((35, 199), self.batch_size, self.model, self.device)
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

    def train(self, cross_validation_enabled: bool = False, train_val_ratio: float = 0.8):
        print_blue(f'Start training for {self.epochs} epochs with model {self.model}, cross validation enabled: {cross_validation_enabled}')
        print_blue(f'Using {train_val_ratio} of total files for training with batch size {self.batch_size}')

        noise_files = get_df_dynamic_noise()
        signal_files = get_df_signal()

        np.random.shuffle(noise_files)
        np.random.shuffle(signal_files)

        eval_len_noise = (1 - train_val_ratio) * len(noise_files)
        eval_len_signal = (1 - train_val_ratio) * len(signal_files)
        num_training_loops = int(1 / (1 - train_val_ratio)) if cross_validation_enabled else 1

        all_max_accuracies = {}
        all_accuracies = {}
        for c in range(num_training_loops):
            print_blue(f'Training loop {c}')
            # prepare noise for doing cross validation
            noise_files_train, noise_files_eval = self._split_train_eval(c, eval_len_noise, noise_files)
            signal_files_train, signal_files_eval = self._split_train_eval(c, eval_len_signal, signal_files)

            dataset_train = self.dataset_class(
                len(signal_files_train),
                noise_files_train,
                signal_files_train,
                signal_strength_upper=self.signal_strength_upper,
                signal_strength_lower=self.signal_strength_lower
            )

            dataset_eval = self.dataset_class(
                len(signal_files_eval),
                noise_files_eval,
                signal_files_eval,
                signal_strength_upper=self.signal_strength_upper,
                signal_strength_lower=self.signal_strength_lower
            )

            dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, drop_last=True)
            dataloader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=self.batch_size, drop_last=True)

            max_acc, accs = self._train_loop(dataloader_train=dataloader_train, dataloader_eval=dataloader_eval)
            all_max_accuracies[c] = max_acc.item()
            all_accuracies[c] = [a.item() for a in accs]
        return all_max_accuracies, all_accuracies

    def _split_train_eval(self, cross_validation_index, num_eval_files, all_files):
        start_index_eval = int(cross_validation_index * num_eval_files)
        end_index_eval = int(cross_validation_index * num_eval_files + num_eval_files)
        
        train_indices = list(crange(end_index_eval, start_index_eval, len(all_files)))
        files_train = np.array(all_files)[train_indices]

        eval_indices = list(range(start_index_eval, end_index_eval))  # should work without %
        files_eval = np.array(all_files)[eval_indices]

        return files_train, files_eval

    def _train_loop(self, dataloader_train, dataloader_eval):
        model = self.get_model()
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)

        max_accuracy = 0
        accuracies = []
        for epoch in range(self.epochs):
            print_green(f'Training Epoch {epoch}')
            for step, (X, y, _) in enumerate(tqdm(dataloader_train, desc='Train', colour='#6ea62e')):
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


def ratio_experiments():
    t = datetime.now().strftime('%Y-%m-%d-%H-%M')
    for r in [0.7, 0.8, 0.9]:
        max_accs, accs = GOPlainCNNTrainer(logging=True, 
                        signal_strength_upper=0.17, 
                        epochs=17, 
                        lr=0.000139, 
                        max_grad_norm=7.639,
                        model='inception_v4').train(train_val_ratio=r, cross_validation_enabled=True)
        file_path = path.join(PATH_TO_CACHE_FOLDER, f'ratio_{r}_{t}.json')
        final_dict = {'max_accs': max_accs, 'accs': accs}
        print_green(f'Writing to file {file_path}')
        with open(file_path, 'w') as file:
            json.dump(final_dict, file)


def signal_strength_experiments():
    file_path = path.join(PATH_TO_CACHE_FOLDER, f'signal_strengths_{datetime.now().strftime("%Y-%m-%d-%H-%M")}.json')
    for f in np.linspace(0.00001, 0.17, 10):
        max_accuracy_dict, accuracies_dict = GOPlainCNNTrainer(logging=True, 
                        signal_strength_upper=f,
                        signal_strength_lower=f ,
                        epochs=1, 
                        lr=0.000139, 
                        max_grad_norm=7.639,
                        model='inception_v4').train(cross_validation_enabled=False)
        print_red(max_accuracy_dict)
        print_red(accuracies_dict)
        max_accuracy = max_accuracy_dict[0]
        accuracies = accuracies_dict[0]
        print_green(f'Writing to file {file_path}')
        with open(file_path, 'a') as file:
            file.write(f'{f},{max_accuracy},{",".join([str(x) for x in accuracies])}\n')

if __name__ == '__main__':
    # ratio_experiments()
    signal_strength_experiments()
    # for f in np.linspace(0.21, 0.17, 5):
    #     max_accuracy, accuracies = GOPlainCNNTrainer(logging=False, signal_strength=f).train()
    #     with open(file_path, 'a') as file:
    #         file.write(f'{f},{max_accuracy},{",".join([str(x.numpy()) for x in accuracies])}\n')