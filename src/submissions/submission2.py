#!pip install timm

import os
import torch

os.chdir('/kaggle/input/go-one/gravitational_oceans')

from src.helper.utils import print_blue, print_green, print_red, print_yellow, PATH_TO_MODEL_FOLDER, PATH_TO_TRAIN_FOLDER, PATH_TO_TEST_FOLDER
from src.data_management.better_crnn_dataset import GOBetterCRNNDataset
import src.ai_nets.cnn_predicter as GOCNNPredictor
from src.ai_nets.pretrained_efficientnet import get_capped_model
from src.ai_nets.transformer import predict, GOTransformer

# params
batch_size = 8
sequence_length = 32

model_params = {
    'num_tokens': 4,
    'dim_model': 2048,
    'num_heads': 2,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'dropout_p': 0.1
}

device = "cuda" if torch.cuda.is_available() else "cpu"

ALL_FILES = [os.path.join(PATH_TO_TRAIN_FOLDER, file_name) for file_name in os.listdir(PATH_TO_TRAIN_FOLDER)]
ALL_FILES += [os.path.join(PATH_TO_TEST_FOLDER, file_name) for file_name in os.listdir(PATH_TO_TEST_FOLDER)]

cnn_model = get_capped_model(os.path.join(PATH_TO_MODEL_FOLDER, 'model_best.pth'))
cnn_model.to(device)
GOCNNPredictor.predict(cnn_model, ALL_FILES)

dataset = GOBetterCRNNDataset(sequence_length=sequence_length)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

transformer_model = torch.load()
model = GOTransformer()
model.load_state_dict(torch.load(os.path.join(PATH_TO_MODEL_FOLDER, 'transformer_best.pt')))
model.to(device)
model.eval()

res = {}

for x, y in dataloader:
    predicted_batch = predict(transformer_model, x, y)
    # cheaty
    target_batch = dataset.get_last_accessed_files()
    dataset.reset_last_accessed_files()

    print_blue(torch.tensor(target_batch).shape)
    print_yellow(torch.tensor(predicted_batch).shape)
    for i in range(batch_size):
        for j in range(sequence_length):
            target = os.path.basename(target_batch[i][j])[:-5]

            if target in res:
                res[target] += [predicted_batch]
            else:
                res[target] = [predicted_batch]
    
print_green(res)



