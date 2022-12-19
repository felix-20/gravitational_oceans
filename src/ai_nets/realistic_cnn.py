# https://www.kaggle.com/code/vslaykovsky/g2net-pytorch-generated-realistic-noise/notebook?scriptVersionId=113484252

import datetime
import torch
import timm

from os import cpu_count, path
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from torch.tensorboard import SummaryWriter

from src.ai_nets.trainer import GOTrainer
from src.data_management.dataset.realistic_dataset import GORealisticNoiseDataset
from src.helper.utils import get_df_noise, get_df_signal, PATH_TO_LOG_FOLDER

class GORealisticCNNTrainer(GOTrainer):
    
    def __init__(self, 
                 df_noise,
                 df_signal,
                 epochs: int = 5,
                 batch_size: int = 32,
                 dropout: float = 0.25,
                 max_grad_norm: float = 1.36, 
                 lr: float = 0.00056, 
                 n_folds: int = 5,
                 fold: int = 0,
                 one_cycle_pct_start: float = 0.1,
                 one_cycle: bool = True,
                 model: str = 'inception_v4') -> None:

        self.epochs = epochs
        self.batch_size = batch_size

        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.n_folds = n_folds
        self.one_cycle = one_cycle
        self.df_noise = df_noise
        self.df_signal = df_signal
        self.dropout = dropout
        self.one_cycle_pct_start = one_cycle_pct_start
        self.model = model
        self.fold = fold

        self.writer = SummaryWriter(path.join(PATH_TO_LOG_FOLDER, 'runs', f'realistic_cnn_{str(datetime.now())}'))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        print(f'Training on {self.device}')

    def evaluate(self, model, dl_eval, return_X=False):
        with torch.no_grad():
            model.eval()
            pred = []
            target = []
            ss = []
            signal_strength = []
            Xs = []
            for X, y, ss in tqdm(dl_eval, desc='Eval'):
                pred.append(model(X.to(self.device)).cpu().squeeze())
                target.append(y)
                signal_strength.append(ss)
                if return_X:
                    Xs.append(X)
            pred = torch.concat(pred)
            target = torch.concat(target)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target.float(), reduction='none').median().item() # Avoiding outlier loss with median
            pred = torch.sigmoid(pred)
            ret = [roc_auc_score(target, pred), loss, pred, target, torch.concat(signal_strength).numpy()]
            if return_X:
                ret.append(torch.concat(Xs).numpy())
            return ret

    def train(self):
        
        dl_train, dl_eval = self.get_dl()

        model = timm.create_model(self.model, pretrained=True, num_classes=1, in_chans=2, drop_rate=self.dropout).to(self.device)
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = None
        if self.one_cycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optim, max_lr=self.lr, total_steps=len(dl_train) * self.epochs, pct_start=self.one_cycle_pct_start)

        max_auc = 0
        for epoch in range(self.epochs):
            for step, (X, y, ss) in enumerate(tqdm(dl_train, desc='Train')):
                pred = model(X.to(self.device)).squeeze()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, y.float().to(self.device))

                optim.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optim.step()
                if scheduler:
                    scheduler.step()
                
                self.writer.add_scalar(f'epoch_{epoch}/loss', loss.item(), step)
                self.writer.add_scalar(f'epoch_{epoch}/lr', scheduler.get_last_lr()[0] if scheduler else self.lr, step)
                self.writer.add_scalar(f'epoch_{epoch}/grad_norm', norm, step)
                self.writer.add_scalar(f'epoch_{epoch}/logit', pred.mean().item(), step)

            auc, loss = self.evaluate(model, dl_eval)[:2]
            if auc > max_auc:
                torch.save(model.state_dict(), f'models/model-f{self.fold}.tph')
                max_auc = auc
            
            self.writer.add_scalar('val/loss', loss, epoch)
            self.writer.add_scalar('val/auc', auc, epoch)
            self.writer.add_scalar('val/max_auc', max_auc, epoch)

        return max_auc


    def get_dl(self):
        kfold = KFold(self.n_folds, shuffle=True, random_state=42)
        df_noise_train, df_noise_eval = None, None
        for f, (train_idx, eval_idx) in enumerate(kfold.split(self.df_noise)):
            if f == self.fold:
                df_noise_train = self.df_noise.loc[train_idx]
                df_noise_eval = self.df_noise.loc[eval_idx]

        df_signal_train, df_signal_eval = None, None
        for f, (train_idx, eval_idx) in enumerate(kfold.split(self.df_signal)):
            if f == self.fold:
                df_signal_train = self.df_signal.loc[train_idx]
                df_signal_eval = self.df_signal.loc[eval_idx]

        ds_train = GORealisticNoiseDataset(
            len(df_signal_train), 
            df_noise_train,
            df_signal_train,
            is_train=True
        )

        ds_eval = GORealisticNoiseDataset(
            len(df_signal_eval), 
            df_noise_eval,
            df_signal_eval
        )

        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=self.batch_size, num_workers=cpu_count(), pin_memory=True)
        dl_eval = torch.utils.data.DataLoader(ds_eval, batch_size=self.batch_size, num_workers=cpu_count(), pin_memory=True)
        return dl_train, dl_eval
    

if __name__ == '__main__':
    GORealisticCNNTrainer(get_df_noise(), get_df_signal()).train()