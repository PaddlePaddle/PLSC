# static
TRAINER_IP_LIST=127.0.0.1

modes=("static" "dynamic")
dtypes=("FP16" "FP32")
data_formats=("NHWC" "NCHW")
gpus=("0" "0,1,2,3")
sample_ratios=(0.1 1.0)

for mode in "${modes[@]}"
do
    if [ $mode = "static" ]; then
        is_static=True
    else
        is_static=False
    fi
    
    for data_format in "${data_formats[@]}"
    do
        for dtype in "${dtypes[@]}"
        do
            if [ $dtype = "FP16" ]; then
                is_fp16=True
            else
                is_fp16=False
            fi

            for sample_ratio in "${sample_ratios[@]}"
            do
                for gpu in "${gpus[@]}"
                do
                    CUDA_VISIBLE_DEVICES=${gpu}
                    num_gpu=`expr ${#gpu} / 2 + 1`
                    python -m paddle.distributed.launch --ips=$TRAINER_IP_LIST --gpus=$CUDA_VISIBLE_DEVICES tools/train.py \
                        --config_file configs/ms1mv3_r50.py \
                        --is_static ${is_static} \
                        --sample_ratio ${sample_ratio} \
                        --batch_size 128 \
                        --dataset MS1M_v3 \
                        --num_classes 93431 \
                        --data_dir MS1M_v3/ \
                        --label_file MS1M_v3/label.txt \
                        --validation_interval_step 100 \
                        --log_interval_step 10 \
                        --fp16 ${is_fp16} \
                        --seed 0 \
                        --train_unit 'step' \
                        --train_num 200 \
                        --decay_boundaries "100,150,180" \
                        --data_format ${data_format} \
                        --val_targets 'agedb_30' \
                        --output Res50_${mode}_${sample_ratio}_${dtype}_${data_format}_gpus${num_gpu}
                done
            done
        done
    done
done
