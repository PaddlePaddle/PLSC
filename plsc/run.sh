#!/usr/bin/env bash

export FLAGS_cudnn_exhaustive_search=true 
export FLAGS_fraction_of_gpu_memory_to_use=0.96
export FLAGS_eager_delete_tensor_gb=0.0
selected_gpus="0,1,2,3,4,5,6,7"
#selected_gpus="4,5,6"

python -m paddle.distributed.launch \
  --selected_gpus $selected_gpus \
  --log_dir mylog \
  do_train.py \
  --model=ResNet_ARCFACE50 \
  --loss_type=dist_softmax \
  --model_save_dir=output \
  --margin=0.5 \
  --train_batch_size 32 \
  --class_dim 85742 \
  --with_test=True
