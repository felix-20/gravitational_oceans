import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from src.data_management.datasets.realistic_dataset import GORealisticNoiseDataset


class GOTransformerDataset(GORealisticNoiseDataset):

    def get_transforms(self):
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.eval()
        config = resolve_data_config({}, model=model)
        return create_transform(**config)
