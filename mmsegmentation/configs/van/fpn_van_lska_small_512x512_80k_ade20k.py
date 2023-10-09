_base_ = [
    '../_base_/models/fpn_van.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py'
]


model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='VAN',
        arch='small',
        k_size=23,
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint='pretrained_model/van/20220603-064559-van_small-224-LSK_DW5_DW7_Small_model_best.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]),
    decode_head=dict(num_classes=150))


gpu_multiples = 1  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer
optimizer = dict(type='AdamW', lr=0.0001*gpu_multiples, weight_decay=0.0001)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=0.0, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=80000//gpu_multiples)
checkpoint_config = dict(by_epoch=False, interval=8000//gpu_multiples)
evaluation = dict(interval=8000//gpu_multiples, metric='mIoU')
data = dict(samples_per_gpu=4)