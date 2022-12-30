# https://www.kaggle.com/code/vslaykovsky/g2net-pytorch-generated-realistic-noise/notebook?scriptVersionId=113484252

from datetime import datetime
from os import cpu_count, path

import timm
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

from src.ai_nets.trainer import GOTrainer
from src.data_management.datasets.realistic_dataset import GORealisticNoiseDataset
from src.helper.utils import get_df_dynamic_noise, get_df_signal, PATH_TO_LOG_FOLDER, PATH_TO_MODEL_FOLDER, print_blue

class GORealisticCNNTrainer(GOTrainer):

    """ found with optuna for dynamic noise
    Trial 186 finished with value: 0.9148327726103305 and parameters: 
    {
        'LR': 0.0001393672397380405, 
        'DROPOUT': 0.1, 
        'MAX_GRAD_NORM': 7.639295602717996, 
        'EPOCHS': 17.0, 
        'GAUSSIAN_NOISE': 1.0, 
        'ONE_CYCLE_PCT_START': 0.0, 
        'MODEL': 'inception_v4', 
        'ONE_CYCLE': False
    }. Best is trial 186 with value: 0.9148327726103305.
    """

    """ found with optuna for static noise
    Trial 16 finished with value: 0.8343663095717244 and parameters:
    {
        'LR': 0.00023710697312064318,
        'DROPOUT': 0.0,
        'MAX_GRAD_NORM': 8.947457257778709,
        'EPOCHS': 20.0,
        'GAUSSIAN_NOISE': 0.0,
        'ONE_CYCLE_PCT_START': 0.0,
        'MODEL': 'efficientnetv2_rw_s',
        'ONE_CYCLE': False
    }. Best is trial 16 with value: 0.8343663095717244.
    """
    
    def __init__(self, 
                 df_noise,
                 df_signal,
                 epochs: int = 20,
                 batch_size: int = 32,
                 dropout: float = 0.0,
                 lr: float = 0.00023710697312064318,
                 max_grad_norm: float = 8.947457257778709, 
                 folds: int = 2, # min 1
                 one_cycle_pct_start: float = 0.0,
                 one_cycle: bool = False,
                 model: str = 'efficientnetv2_rw_s',
                 gaussian_noise: float = 0.0,
                 logging: bool = True,
                 dataset_class = GORealisticNoiseDataset) -> None:

        self.epochs = epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        self.one_cycle = one_cycle
        self.df_noise = df_noise
        self.df_signal = df_signal
        self.dropout = dropout
        self.one_cycle_pct_start = one_cycle_pct_start
        self.model = model
        self.folds = folds
        self.gaussian_noise = gaussian_noise
        self.logging = logging
        self.dataset_class = dataset_class

        if logging:
            self.writer = SummaryWriter(path.join(PATH_TO_LOG_FOLDER, 'runs', f'best_static_realistic_cnn_{str(datetime.now())}'))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert self.folds > 0

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
        result_max = 0
        for fold in range(self.folds):
            print_blue(f'----------------------------- FOLD {fold} -----------------------------')

            dl_train, dl_eval = self.get_dl(fold)

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
                    
                    if self.logging:
                        self.writer.add_scalar(f'loss/epoch_{epoch}', loss.item(), step)
                        self.writer.add_scalar(f'lr/epoch_{epoch}', scheduler.get_last_lr()[0] if scheduler else self.lr, step)
                        self.writer.add_scalar(f'grad_norm/epoch_{epoch}', norm, step)
                        self.writer.add_scalar(f'logit/epoch_{epoch}', pred.mean().item(), step)

                auc, loss = self.evaluate(model, dl_eval)[:2]
                if auc > max_auc:
                    torch.save(model.state_dict(), f'{PATH_TO_MODEL_FOLDER}/best_static_model-f{fold}.tph')
                    max_auc = auc
                
                if self.logging:
                    self.writer.add_scalar('val/loss', loss, epoch)
                    self.writer.add_scalar('val/auc', auc, epoch)
                    self.writer.add_scalar('val/max_auc', max_auc, epoch)

                if epoch > 5:
                    result_max = max(result_max, max_auc)
        
        return result_max

    def get_dl(self, fold):
        kfold = KFold(max(2, self.folds), shuffle=True, random_state=42)
        df_noise_train, df_noise_eval = None, None
        for f, (train_idx, eval_idx) in enumerate(kfold.split(self.df_noise)):
            if f == fold:
                df_noise_train = self.df_noise.loc[train_idx]
                df_noise_eval = self.df_noise.loc[eval_idx]

        df_signal_train, df_signal_eval = None, None
        for f, (train_idx, eval_idx) in enumerate(kfold.split(self.df_signal)):
            if f == fold:
                df_signal_train = self.df_signal.loc[train_idx]
                df_signal_eval = self.df_signal.loc[eval_idx]

        ds_train = self.dataset_class(
            len(df_signal_train),
            df_noise_train,
            df_signal_train,
            is_train=True,
            gaussian_noise=self.gaussian_noise
        )

        ds_eval = self.dataset_class(
            len(df_signal_eval),
            df_noise_eval,
            df_signal_eval,
            gaussian_noise=self.gaussian_noise
        )

        dl_train = torch.utils.data.DataLoader(ds_train, batch_size=self.batch_size, num_workers=cpu_count(), pin_memory=True)
        dl_eval = torch.utils.data.DataLoader(ds_eval, batch_size=self.batch_size, num_workers=cpu_count(), pin_memory=True)
        return dl_train, dl_eval


if __name__ == '__main__':
    GORealisticCNNTrainer(get_df_dynamic_noise(), get_df_signal()).train()
