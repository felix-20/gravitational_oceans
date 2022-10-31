import torch
from torch.utils.data import DataLoader
import torchvision.models as models

from dataset import GODataset
from utils import print_green

classes = ['no_cw', 'cw']


def test(model, batch_size):
    print('begin testing')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    print('setting up dataloader')
    test_dataset = GODataset('./data')
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('getting ground truth data')
    dataiter = iter(test_loader)
    inputs, labels = next(dataiter)

    inputs = inputs.to(device).float()
    labels = labels.to(device)

    # print images
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    output = model(inputs)
    tmp, predicted = torch.max(output, 1)

    print_green(tmp)
    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))

if __name__ == "__main__":
    path = './models/efficientnet_2022-10-31_18:19.pt'
    efficientnet = models.efficientnet_b0()
    efficientnet.load_state_dict(torch.load(path))
    efficientnet.eval()

    test(efficientnet, 4)