MODEL=van_tiny # van_{tiny, small, base}
DROP_PATH=0.1 # drop path rates [0.1, 0.1, 0.1] for [tiny, small, base]
      

CUDA_VISIBLE_DEVICES=0,1 bash distributed_train.sh 2 29515 /data4/DATA/imagenet1k \
	  --model $MODEL -b 64 --lr 1e-3 --drop-path $DROP_PATH --k_size 7 \
	  --log_name LSKA_K7

