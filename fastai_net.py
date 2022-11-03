from fastai.vision.all import *
from pathlib import Path

from utils import PATH_TO_TRAIN_FOLDER

DATASET_PATH = Path(f'{PATH_TO_TRAIN_FOLDER}/images/')

mask_datablock = DataBlock(
    get_items=get_image_files,
    get_y=parent_label,
    blocks=(ImageBlock, CategoryBlock), 
    #item_tfms=RandomResizedCrop(224, min_scale=0.3),
    splitter=RandomSplitter(valid_pct=0.2, seed=100),
    #batch_tfms=aug_transforms(mult=2)
)

print(mask_datablock)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

dls = mask_datablock.dataloaders(
    DATASET_PATH,
    device=device
    )
print(dls)

learn = vision_learner(dls, efficientnet_b0, metrics=error_rate, pretrained=False)
learn.fine_tune(10)

#interp = ClassificationInterpretation.from_learner(learn)
#interp.plot_confusion_matrix()