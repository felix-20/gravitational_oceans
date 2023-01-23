from ai_nets.realistic_cnn import GORealisticCNNTrainer
from src.data_management.datasets.transformer_dataset import GOTransformerDataset
from src.helper.utils import get_df_dynamic_noise, get_df_signal

if __name__ == '__main__':
    GORealisticCNNTrainer(get_df_dynamic_noise(), get_df_signal(), dataset_class=GOTransformerDataset).train()
