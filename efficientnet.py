import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from datetime import datetime

from dataset import GODataset
from model_test import test
from utils import print_green, print_red

print('finished imports')

classes = ['no_cw', 'cw']
batch_size = 4

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using {device} for inference')

print('setting up dataloader')

train_dataset = GODataset('./data')
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

print('building efficientnet')

efficientnet = models.efficientnet_b0()
efficientnet.eval().to(device)

print('defining loss and optimizer')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(efficientnet.parameters(), lr=0.001, momentum=0.9)

print('begin training')

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # upload to gpu
        inputs = inputs.to(device).float()
        labels = labels.to(device)

        print_red(labels)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = efficientnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        #item_loss = loss.item()
        #print(f'[{epoch + 1}, {i + 1:5d}] loss: {item_loss}')

print('finished training')

torch.save(efficientnet.state_dict(), f'models/efficientnet_{datetime.now().strftime("%Y-%m-%d_%H:%M")}.pt')

############ TESTING ############

test(efficientnet, batch_size)