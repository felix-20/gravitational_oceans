import h5py
import numpy as np
import torch
import torchvision.models as models

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using {device} for inference')

efficientnet = models.efficientnet_b0(pretrained=False)

efficientnet.eval().to(device)

uris = [
    'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
    'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
]

batch = torch.cat(
    [utils.prepare_input_from_uri(uri) for uri in uris]
).to(device)


with torch.no_grad():
    output = torch.nn.functional.softmax(efficientnet(batch), dim=1)

results = utils.pick_n_best(predictions=output, n=5)
