"""24 full passes over MLT2017 + SynText1 + SynText2 training images.

Use the merged annotation produced by prepare_english_datasets.py. Image names
already contain their dataset-relative prefix, so no extra ``images/`` is added.
The schedule and learning rate intentionally override the generic config's
OCRDINO_EPOCHS/OCRDINO_LR values. CLI --cfg-options remains available.
"""
import os

_base_ = './ocrdino_r50_4scale_ctc.py'

data_root = os.environ.get('OCRDINO_DATA_ROOT', 'data/english')
image_prefix = os.environ.get('OCRDINO_IMAGES', '')
train_json = 'annotations/pretrain.json'
max_epochs = 24
learning_rate = 0.0001

train_dataloader = dict(dataset=dict(
    data_root=data_root, ann_file=train_json,
    data_prefix=dict(img=image_prefix)))
train_cfg = dict(max_epochs=max_epochs, val_interval=max_epochs + 1)
optim_wrapper = dict(optimizer=dict(lr=learning_rate))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(type='MultiStepLR', by_epoch=True, begin=0, end=max_epochs,
         milestones=[19, 22], gamma=0.1)
]

# This is the COCO-to-text initialization, before any OCR training.
load_from = os.environ.get('OCRDINO_PRETRAINED', 'checkpoints/ocrdino_init.pth')
resume = False
work_dir = os.environ.get(
    'OCRDINO_PRETRAIN_WORK_DIR', 'work_dirs/ocrdino_pretrain24')
