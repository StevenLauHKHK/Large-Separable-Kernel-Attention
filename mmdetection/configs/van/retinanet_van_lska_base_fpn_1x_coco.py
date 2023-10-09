_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        _delete_=True,
        type='VAN',
        arch='base',
        k_size=23,
        out_indices=(0,1,2,3),
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint='pretrained_model/van/20221027-101059-van_small-224-Double_LSK_DW3_DW3_w_dilation_2_Small_model_best.pth')),
    neck=dict(in_channels=[64, 128, 320, 512], start_level=1, num_outs=5))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
)

lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(max_epochs=12)