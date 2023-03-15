from os import path

import timm
import torch
import cv2

from src.helper.utils import PATH_TO_SOURCE_FOLDER, print_blue, print_green, print_red, print_yellow, normalize_image


class GOHadamardLayer(torch.nn.Module):
    def __init__(self, 
                 in_features, 
                 device, 
                 path_to_weight_image=path.join(PATH_TO_SOURCE_FOLDER, 'ai_nets', 'pretrained_weights', 'signals_pretrained.png')):
        super().__init__()

        weight_data = cv2.imread(path_to_weight_image, cv2.IMREAD_GRAYSCALE) / 255.0
        tensor = torch.tensor(weight_data, device=device, dtype=torch.float)
        self.weight = torch.nn.Parameter(tensor)
    
    def forward(self, x):
        return torch.mul(self.weight, x)


class GODenseMaxPoolModel(torch.nn.Module):
    def __init__(self, input_shape, batch_size, model, device):
        super().__init__()
        self.weighting = GOHadamardLayer(in_features=input_shape, device=device)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=7, stride=2, padding=0, dilation=3)
        # torch.nn.Linear(in_features=np.prod(input_shape), out_features=np.prod(input_shape), device=device)
        self.model = timm.create_model(model, num_classes=1, in_chans=2).to(device)

        self.batch_size = batch_size
        # self.input_shape = input_shape
        self.height = input_shape[-1]
        self.width = input_shape[-2]
    
    def forward(self, X): # shape (batch_size, detectors, height, width)
        global epoch_index, epoch_changed
        weighted = self.weighting(X)
        
        #linearized = pooled.reshape(self.batch_size, 2, -1)
        #weighted = self.weighting(linearized)
        #weighted = weighted.reshape(self.batch_size, 2, self.width, self.height)
        pooled = self.max_pool(weighted)
        
        if epoch_changed:
            epoch_changed = False
            cv2.imwrite(f'./gravitational_oceans/tmp/weights_{epoch_index}.png', normalize_image(self.weighting.weight.cpu().detach().numpy()))
            print_red('saved')
        
        return self.model(pooled)