_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='VAN',
        arch='base',
        k_size=23,
        out_indices=(0,1,2,3),
        drop_path_rate=0.1,
        init_cfg=dict(type='Pretrained', checkpoint='pretrained_model/van/20221223-161804-van_base-224-Double_LKA_VAN_DW5_DW11_w_dilation_3_Base.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
)
lr_config = dict(warmup_iters=1000, step=[8, 11])
runner = dict(max_epochs=12)
