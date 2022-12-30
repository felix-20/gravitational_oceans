import optuna

from src.ai_nets.realistic_cnn import GORealisticCNNTrainer
from src.helper.utils import get_df_dynamic_noise, get_df_signal, get_df_static_noise, print_red


def objective(trial: optuna.Trial):
    LR = trial.suggest_float('LR', 0.0001, 0.005, log=True)
    DROPOUT = trial.suggest_categorical('DROPOUT', [0., 0.1, 0.25, 0.5])
    MAX_GRAD_NORM = trial.suggest_float('MAX_GRAD_NORM', 1, 20)
    EPOCHS = int(trial.suggest_float('EPOCHS', 10, 20, step=1.))
    GAUSSIAN_NOISE = trial.suggest_categorical('GAUSSIAN_NOISE', [0., 1.])
    ONE_CYCLE_PCT_START = trial.suggest_categorical('ONE_CYCLE_PCT_START', [0., 0.1])
    MODEL = trial.suggest_categorical('MODEL', ['resnext50_32x4d', 'efficientnetv2_rw_s', 'seresnext50_32x4d', 'inception_v4'])
    ONE_CYCLE = trial.suggest_categorical('ONE_CYCLE', [True, False])

    print(EPOCHS)

    trainer = GORealisticCNNTrainer(
        get_df_static_noise(),
        get_df_signal(),
        lr=LR,
        dropout=DROPOUT,
        max_grad_norm=MAX_GRAD_NORM,
        epochs=EPOCHS,
        gaussian_noise=GAUSSIAN_NOISE,
        one_cycle_pct_start=ONE_CYCLE_PCT_START,
        model=MODEL,
        one_cycle=ONE_CYCLE,
        logging=False
    )

    return trainer.train()


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=300)
print_red(study.best_params)
