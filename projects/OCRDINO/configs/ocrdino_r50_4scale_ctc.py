"""Shared-backbone DINO text detection and lightweight CTC recognition.

Run from the repository root. Environment variables make server paths explicit;
all training images are used and no validation loop is configured by default.
"""
import os

_base_ = '../../../configs/dino/dino-4scale_r50_improved_8xb2-12e_coco.py'

custom_imports = dict(
    imports=['projects.OCRDINO.ocrdino'], allow_failed_imports=False)

data_root = os.environ.get('OCRDINO_DATA_ROOT', 'data/ocr')
train_json = os.environ.get('OCRDINO_TRAIN_JSON', 'train.json')
image_prefix = os.environ.get('OCRDINO_IMAGES', 'images/')
dictionary = os.environ.get(
    'OCRDINO_DICTIONARY', 'projects/OCRDINO/dictionaries/english.json')
batch_size = int(os.environ.get('OCRDINO_BATCH_SIZE', '1'))
accumulative_counts = int(os.environ.get('OCRDINO_ACCUMULATION', '4'))
num_workers = int(os.environ.get('OCRDINO_WORKERS', '4'))
max_epochs = int(os.environ.get('OCRDINO_EPOCHS', '30'))
learning_rate = float(os.environ.get('OCRDINO_LR', '0.0001'))

model = dict(
    type='OCRDINO',
    num_queries=900,
    data_preprocessor=dict(pad_size_divisor=32),
    # C2 is used by recognition only. The inherited neck still receives C3-C5.
    backbone=dict(
        out_indices=(0, 1, 2, 3), frozen_stages=-1,
        with_cp=True, init_cfg=None),
    bbox_head=dict(type='OCRDINOHead', num_classes=1),
    recognition_head=dict(
        type='LightCTCHead',
        dictionary=dictionary,
        in_channels=256,
        hidden_channels=128,
        lstm_hidden_size=128,
        roi_height=8,
        width_buckets=(32, 64, 128),
        width_scale=16.0,
        feature_stride=4,
        sampling_ratio=2,
        gn_groups=32,
        loss_weight=1.0),
    dn_max_queries=128,
    dn_cfg=dict(group_cfg=dict(num_dn_queries=128)),
    rec_train_cfg=dict(
        max_rois=128,
        pred_box_prob=0.0,
        pred_box_start_epoch=5,
        min_pred_iou=0.7),
    rec_test_cfg=dict(score_thr=0.3, max_per_img=100),
    test_cfg=dict(max_per_img=300))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='OCRLoadAnnotations', with_bbox=True),
    # Do not mirror text or crop away characters while retaining full labels.
    dict(type='Resize', scale=(1280, 800), keep_ratio=True),
    dict(type='OCRPackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1280, 800), keep_ratio=True),
    dict(
        type='OCRPackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    _delete_=True,
    batch_size=batch_size,
    num_workers=num_workers,
    persistent_workers=num_workers > 0,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='OCRTextDataset',
        data_root=data_root,
        ann_file=train_json,
        data_prefix=dict(img=image_prefix),
        dictionary=dictionary,
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline))

# The full training set is retained. The final epoch is not a validated best.
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs,
    val_interval=max_epochs + 1)
val_dataloader = None
val_cfg = None
val_evaluator = None
test_dataloader = None
test_cfg = None
test_evaluator = None

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='float16',
    loss_scale='dynamic',
    accumulative_counts=accumulative_counts,
    optimizer=dict(type='AdamW', lr=learning_rate, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR', by_epoch=True, begin=0, end=max_epochs,
        milestones=sorted(set([max(1, int(max_epochs * 0.8)),
                               max(1, int(max_epochs * 0.9))])), gamma=0.1)
]
default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=3,
        save_last=True, save_best=None))
custom_hooks = [dict(type='OCRDictionaryHook')]
auto_scale_lr = dict(enable=False, base_batch_size=16)
# ResNet's built-in checkpointing uses reentrant checkpoint(). All trainable
# parameters participate, including explicit zero losses on empty local ranks.
find_unused_parameters = False
randomness = dict(seed=3407, deterministic=False)

# Must point to the converted initialization checkpoint, not the 80-class file.
load_from = os.environ.get('OCRDINO_PRETRAINED', 'checkpoints/ocrdino_init.pth')
resume = False
work_dir = os.environ.get('OCRDINO_WORK_DIR', 'work_dirs/ocrdino_ctc_v1')
