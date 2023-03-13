import optuna

from src.ai_nets.plain_cnn import GOPlainCNNTrainer
from src.helper.utils import print_red


def objective(trial: optuna.Trial):
    LR = trial.suggest_float('LR', 0.0001, 0.005, log=True)
    MAX_GRAD_NORM = trial.suggest_float('MAX_GRAD_NORM', 1, 20)
    EPOCHS = int(trial.suggest_float('EPOCHS', 10, 20, step=1.))
    MODEL = trial.suggest_categorical('MODEL', ['resnext50_32x4d', 'efficientnetv2_rw_s', 'seresnext50_32x4d', 'inception_v4'])


    trainer = GOPlainCNNTrainer(
        epochs=EPOCHS,
        lr=LR,
        max_grad_norm=MAX_GRAD_NORM,
        model=MODEL,
        logging=False,
        signal_strength=0.17
    )

    return trainer.train()


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=300)
print_red(study.best_params)