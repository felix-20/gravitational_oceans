# https://www.kaggle.com/code/laeyoung/g2net-large-kernel-inference/notebook?scriptVersionId=111267004

import os

if 'IS_CHARLIE' in os.environ:
    print('We are on Charlie')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import gc
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from timm import create_model

from src.helper.utils import PATH_TO_MODEL_FOLDER, PATH_TO_TEST_FOLDER, print_blue, print_green, print_red, print_yellow


def normalize(X):
    X = (X[..., None].view(X.real.dtype) ** 2).sum(-1)
    POS = int(X.size * 0.99903)
    EXP = norm.ppf((POS + 0.4) / (X.size + 0.215))
    scale = np.partition(X.flatten(), POS, -1)[POS]
    X /= scale / EXP.astype(scale.dtype) ** 2
    return X

def dataload(filepath):
    astime = np.full([2, 360, 5760], np.nan, dtype=np.float32)
    with h5py.File(filepath, 'r') as f:
        fid, _ = os.path.splitext(os.path.split(filepath)[1])
        HT = (np.asarray(f[fid]['H1']['timestamps_GPS']) / 1800).round().astype(np.int64)
        LT = (np.asarray(f[fid]['L1']['timestamps_GPS']) / 1800).round().astype(np.int64)
        MIN = min(HT.min(), LT.min()); HT -= MIN; LT -= MIN
        H1 = normalize(np.asarray(f[fid]['H1']['SFTs'], np.complex128))
        valid = HT < 5760; astime[0][:, HT[valid]] = H1[:, valid]
        L1 = normalize(np.asarray(f[fid]['L1']['SFTs'], np.complex128))
        valid = LT < 5760; astime[1][:, LT[valid]] = L1[:, valid]
    gc.collect()
    return fid, astime, H1.mean(), L1.mean()

class LargeKernel_debias(nn.Conv2d):
    def forward(self, input: torch.Tensor):
        finput = input.flatten(0, 1)[:, None]
        target = abs(self.weight)
        target = target / target.sum((-1, -2), True)
        joined_kernel = torch.cat([self.weight, target], 0)
        reals = target.new_zeros(
            [1, 1] + [s + p * 2 for p, s in zip(self.padding, input.shape[-2:])]
        )
        reals[
            [slice(None)] * 2 + [slice(p, -p) if p != 0 else slice(None) for p in self.padding]
        ].fill_(1)
        output, power = torch.nn.functional.conv2d(
            finput, joined_kernel, padding=self.padding
        ).chunk(2, 1)
        ratio = torch.div(*torch.nn.functional.conv2d(reals, joined_kernel).chunk(2, 1))
        output.sub_(power.mul_(ratio))
        return output.unflatten(0, input.shape[:2]).flatten(1, 2)

def preprocess(num, input, H1, L1):
    input = torch.from_numpy(input).to('cuda', non_blocking=True)
    rescale = torch.tensor([[H1, L1]]).to('cuda', non_blocking=True)
    tta = (
        torch.randn(
            [num, *input.shape, 2], device=input.device, dtype=torch.float32
        )
        .square_()
        .sum(-1)
    )
    tta *= rescale[..., None, None] / 2
    valid = ~torch.isnan(input); tta[:, valid] = input[valid].float()
    return tta

def get_model(path):
    model = create_model(
        'tf_efficientnetv2_b0',
        in_chans=32,
        num_classes=2,
    )
    state_dict = torch.load(path)
    C, _, H, W = state_dict['conv_stem.2.weight'].shape
    model.conv_stem = nn.Sequential(
        nn.Identity(),
        nn.AvgPool2d((1, 9), (1, 8), (0, 4), count_include_pad=False),
        LargeKernel_debias(1, C, [H, W], 1, [H//2, W//2], 1, 1, False),
        model.conv_stem,
    )
    model.load_state_dict(state_dict)
    model.cuda().eval()
    return model

@torch.no_grad()
def inference(model, path):
    file_path = glob.glob(os.path.join(path, '*.hdf5'))
    FID, RES = [], []
    with ProcessPoolExecutor(2) as pool:
        for fid, input, H1, L1 in pool.map(dataload, sorted(file_path)):
            tta = preprocess(64, input, H1, L1)
            FID += [fid]
            RES += [model(tta).softmax(-1)[..., 1].mean(0)]
    return FID, torch.stack(RES, 0).cpu().float().numpy()

def get_capped_model(path):
    full_model = get_model(path)
    capped_model = nn.Sequential(*(list(full_model.children())[:-4]))
    return capped_model # output of shape [192, 1280]


if __name__ == '__main__':
    model = get_capped_model(os.path.join(PATH_TO_MODEL_FOLDER, 'model_best.pth'))
    print_red('got model')

    print_yellow('\n'.join([str(i) for i in list(model.children())[-1:]]))

    """
    fid, infer = inference(
        model, os.path.join(PATH_TO_TEST_FOLDER, 'cw_hdf5')
    )
    print_red('after inference')
    result = pd.DataFrame.from_dict({'id': fid, 'target': infer})
    result.to_csv('submission.csv', index=False)
    print_green('DONE')
    """
