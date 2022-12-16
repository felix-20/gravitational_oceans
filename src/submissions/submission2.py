#!pip install timm

import os
import torch
import numpy as np
import csv
from tqdm import tqdm

#os.chdir('/kaggle/input/go-one/gravitational_oceans')

from src.helper.utils import print_blue, print_green, print_red, print_yellow, PATH_TO_MODEL_FOLDER, PATH_TO_TRAIN_FOLDER, PATH_TO_TEST_FOLDER, PATH_TO_CACHE_FOLDER
from src.data_management.better_crnn_dataset import GOBetterCRNNDataset
import src.ai_nets.cnn_predicter as GOCNNPredictor
from src.ai_nets.pretrained_efficientnet import get_capped_model
from src.ai_nets.transformer import GOTransformer, PositionalEncoding, GOTransformerTrainer

model_params = {
    'num_tokens': 4,
    'dim_model': 2048,
    'num_heads': 2,
    'num_encoder_layers': 3,
    'num_decoder_layers': 3,
    'dropout_p': 0.1
}

device = "cuda" if torch.cuda.is_available() else "cpu"

ALL_FILES = [os.path.join(PATH_TO_TEST_FOLDER, file_name) for file_name in os.listdir(PATH_TO_TEST_FOLDER)]
ALL_FILES += [os.path.join(PATH_TO_TRAIN_FOLDER, file_name) for file_name in os.listdir(PATH_TO_TRAIN_FOLDER)]

if not os.path.isdir(os.path.join(PATH_TO_CACHE_FOLDER, 'pre_predicted')):
    cnn_model = get_capped_model(os.path.join(PATH_TO_MODEL_FOLDER, 'model_best.pth'))
    cnn_model.to(device)
    GOCNNPredictor.predict(cnn_model, ALL_FILES)

go_trainer = GOTransformerTrainer(dim_model=2048)
go_trainer.build(model_params)
dataset = GOBetterCRNNDataset(sequence_length=go_trainer.sequence_length)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=go_trainer.batch_size)

transformer_model = torch.load(os.path.join(PATH_TO_MODEL_FOLDER, 'transformer_best.pt'))
transformer_model.to(device)
transformer_model.eval()

samples = {}

for x, y in tqdm(dataloader, 'doing inference'):
    predicted_batch = go_trainer.predict(transformer_model, x, y).cpu()
    # cheaty
    target_batch = dataset.get_last_accessed_files()
    dataset.reset_last_accessed_files()

    #print_blue(len(target_batch), len(target_batch[0]))
    #print_yellow(torch.tensor(predicted_batch).shape)
    for i in range(len(target_batch)):
        for j in range(go_trainer.sequence_length):
            target = os.path.basename(target_batch[i][j])[:-5]

            if target in samples:
                samples[target] += [predicted_batch[i][j].item()]
            else:
                samples[target] = [predicted_batch[i][j].item()]
    
res = {}
for file_id, labels in samples.items():
    res[file_id] = np.mean(labels)

print_green('Inference complete')

print_blue('dumping to file')
with open('submission.csv','w') as f:
    w = csv.writer(f)
    w.writerow(('id', 'target'))
    w.writerows(res.items())


