MODEL=van_tiny # van_{tiny, small, base}

CUDA_VISIBLE_DEVICES=1 python3 validate.py /data4/DATA/imagenet1k --model $MODEL --k_size 23 \
  --checkpoint /data0/steven/LSK/LSK/output/train/20220603-064921-van_tiny-224-LSK_DW5_DW7_Tiny/checkpoint-309.pth.tar -b 128



