import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader

from dataset import GODataset
from utils import print_green

print('finished imports')

classes = ['no_cw', 'cw']

print('setting up dataloader')

train_dataset = GODataset('./data')
train_loader = DataLoader(train_dataset, batch_size=2)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using {device} for inference')

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


######################## TESTING ########################

print('begin testing')

test_dataset = GODataset('./data')
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)


dataiter = iter(test_loader)
inputs, labels = next(dataiter)

inputs = inputs.to(device).float()
labels = labels.to(device)

# print images
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

output = efficientnet(inputs)
tmp, predicted = torch.max(output, 1)

print_green(tmp)
print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(4)))
