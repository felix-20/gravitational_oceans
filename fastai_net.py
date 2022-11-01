from fastai.vision.all import *
from pathlib import Path

DATASET_PATH = Path('./data/images/')

mask_datablock = DataBlock(
    get_items=get_image_files,
    get_y=parent_label,
    blocks=(ImageBlock, CategoryBlock), 
    #item_tfms=RandomResizedCrop(224, min_scale=0.3),
    splitter=RandomSplitter(valid_pct=0.2, seed=100),
    #batch_tfms=aug_transforms(mult=2)
)

print(mask_datablock)

dls = mask_datablock.dataloaders(
    DATASET_PATH,
    device=torch.device('cuda')
    )
print(dls)

learn = vision_learner(dls, resnet50, metrics=error_rate)
learn.fine_tune(10)

#interp = ClassificationInterpretation.from_learner(learn)
#interp.plot_confusion_matrix()