# Large Separable Kernel Attention (LSKA)

This repository implements the model proposed in the paper:

Kin Wai Lau, Lai-Man Po, Yasar Abbas Ur Rehman, **Large Separable Kernel Attention: Rethinking the Large Kernel Attention Design in CNN**

[[arXiv paper]](https://arxiv.org/abs/2309.01439)

The implementation code is based on the **Visual Attention Network (VAN)**, Computational Visual Media, 2023. For more information, please refer to the [link](https://github.com/Visual-Attention-Network/VAN-Classification).

## Citing

When using this code, kindly reference:

```
@article{lau2024large,
  title={Large Separable Kernel Attention: Rethinking the Large Kernel Attention Design in CNN},
  author={Lau, Kin Wai and Po, Lai-Man and Rehman, Yasar Abbas Ur},
  journal={Expert Systems with Applications},
  volume={236},
  pages={121352},
  year={2024},
  publisher={Elsevier}
}
```

## Pretrained models

You can download our pretrained models on ImageNet-1K:

Pretrained ImageNet-1K LSKA (kernel size: 23): [link](https://portland-my.sharepoint.com/:f:/g/personal/kinwailau6-c_my_cityu_edu_hk/Ek_Y2ftu6uJEk_RMvvkEYFoBuSwQcJYf1hz1Y_6P3Qj5Hw?e=f1NjnE).

## Preparation

* Requirements:
```
1. Pytorch >= 1.7
2. timm == 0.4.12
```

### Train 

We use 4 GPUs for training by default.  Run command (It has been writen in train_lska.sh):

```bash
MODEL=van_tiny # van_{tiny, small, base}
DROP_PATH=0.1 # drop path rates [0.1, 0.1, 0.1] for [tiny, small, base]
# Kernel size should be [7, 11, 23, 35, 53]
CUDA_VISIBLE_DEVICES=0,1,2,3 bash distributed_train.sh 4 /path/to/imagenet \
	  --model $MODEL -b 128 --lr 1e-3 --drop-path $DROP_PATH --k_size kernel_size \
	  --log_name log_file_name
```

### Validate

Run command (It has been writen in eval.sh) as:

```bash
MODEL=van_tiny # van_{tiny, small, base}
CUDA_VISIBLE_DEVICES=0 python3 validate.py /path/to/imagenet --model $MODEL --k_size kernel_size \
  --checkpoint /path/to/model -b 128

```
### Object Detection and Segmentation

We also include the configuration file for the object detection and semantic segmentation in this repository. For details, please check folder mmdetection and mmsegmentation.
