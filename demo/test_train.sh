#!/bin/env bash

export PATH=/home/lilong/sandyhouse/PLSC/python/bin:$PATH
export FLAGS_eager_delete_tensor_gb=0.0
export GLOG_v=0

## case 2: run with softmax
#python do_train.py \
#    --model_save_dir="./saved_model" \
#    --data_dir="./data" \
#    --num_epochs=2 \
#    --loss_type='softmax'
#
# case 3: run with distarcface
python -m paddle.distributed.launch \
    --log_dir='mylog' \
    --selected_gpus="0,1,2,3,4,5,6,7" \
    --started_port="12349" \
    do_train.py \
    --data_dir="./data" \
    --num_epochs=10000000 \
    --loss_type='dist_arcface'
#
## case 4: run with distsoftmax
#python -m paddle.distributed.launch \
#    --log_dir='mylog' \
#    --selected_gpus="0,1" \
#    --started_port="12345" \
#    do_train.py \
#    --model_save_dir="./saved_model" \
#    --data_dir="./data" \
#    --num_epochs=2 \
#    --loss_type='dist_softmax'

## case 5: run from checkpoints with same number of trainers
#python -m paddle.distributed.launch \
#    --log_dir='mylog' \
#    --selected_gpus="0,1" \
#    --started_port="12345" \
#    do_train.py \
#    --model_save_dir="./saved_model" \
#    --checkpoint_dir="./saved_model/1" \
#    --data_dir="./data" \
#    --num_epochs=2 \
#    --loss_type='dist_softmax'

## case 6: run from checkpoints with incresement number of trainers
#python -m paddle.distributed.launch \
#    --log_dir='mylog' \
#    --selected_gpus="0,1,2,3" \
#    --started_port="12345" \
#    do_train.py \
#    --model_save_dir="./saved_model" \
#    --checkpoint_dir="./saved_model/0" \
#    --data_dir="./data" \
#    --num_epochs=2 \
#    --loss_type='dist_softmax'
#
## case 7: run from checkpoints with decreasement number of trainers
#python -m paddle.distributed.launch \
#    --log_dir='mylog' \
#    --selected_gpus="0,1" \
#    --started_port="12345" \
#    do_train.py \
#    --model_save_dir="./saved_model" \
#    --checkpoint_dir="./saved_model/0" \
#    --data_dir="./data" \
#    --num_epochs=2 \
#    --loss_type='dist_softmax'

## case 8: save models to hdfs
#python -m paddle.distributed.launch \
#    --log_dir='mylog' \
#    --selected_gpus="0,1" \
#    --started_port="12345" \
#    do_train.py \
#    --model_save_dir="./saved_model" \
#    --data_dir="./data" \
#    --fs_name=${FS_NAME} \
#    --fs_ugi=${FS_UGI} \
#    --fs_dir_save="/user/paddle/lilong/models/saved_model2" \
#    --num_epochs=2 \
#    --loss_type='dist_softmax'

## case 9: get models from hdfs
#python -m paddle.distributed.launch \
#    --log_dir='mylog' \
#    --selected_gpus="0,1" \
#    --started_port="12345" \
#    do_train.py \
#    --checkpoint_dir="./saved_model/" \
#    --data_dir="./data" \
#    --fs_name=${FS_NAME} \
#    --fs_ugi=${FS_UGI} \
#    --fs_dir_load="/user/paddle/lilong/models/saved_model/0" \
#    --num_epochs=2 \
#    --loss_type='dist_softmax'

## case 10: get models from hdfs and save models to hdfs
#python3 -m paddle.distributed.launch \
#    --log_dir='mylog' \
#    --selected_gpus="0,1" \
#    --started_port="12345" \
#    do_train.py \
#    --checkpoint_dir="./saved_model/" \
#    --data_dir="./data" \
#    --fs_name=${FS_NAME} \
#    --fs_ugi=${FS_UGI} \
#    --fs_dir_load="/user/paddle/lilong/models/saved_model/0" \
#    --fs_dir_save="/user/paddle/lilong/models/saved_model2" \
#    --num_epochs=2 \
#    --loss_type='dist_softmax'
