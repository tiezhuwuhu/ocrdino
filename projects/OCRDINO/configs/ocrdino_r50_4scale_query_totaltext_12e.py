"""12 fresh fine-tuning epochs on Total-Text train, after OCR pretraining.

Load all OCR weights, including recognition; do not convert this checkpoint or
resume its optimizer/epoch counter. Total-Text test is not used for training or
validation. Keep the pretraining dictionary and architecture unchanged.
"""
import os

_base_ = './ocrdino_r50_4scale_query.py'

data_root = os.environ.get('OCRDINO_DATA_ROOT', 'data/english')
image_prefix = os.environ.get('OCRDINO_IMAGES', '')
train_json = 'annotations/totaltext_train.json'
max_epochs = 12
learning_rate = 0.00005

train_dataloader = dict(dataset=dict(
    data_root=data_root, ann_file=train_json,
    data_prefix=dict(img=image_prefix)))
train_cfg = dict(max_epochs=max_epochs, val_interval=max_epochs + 1)
optim_wrapper = dict(optimizer=dict(lr=learning_rate))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=max_epochs,
         milestones=[8, 11], gamma=0.1)
]

# Deliberately separate from OCRDINO_PRETRAINED, which still points to COCO init.
load_from = os.environ.get(
    'OCRDINO_FINETUNE_FROM', 'work_dirs/ocrdino_query_pretrain24/epoch_24.pth')
resume = False
work_dir = os.environ.get(
    'OCRDINO_FINETUNE_WORK_DIR', 'work_dirs/ocrdino_query_totaltext12')
