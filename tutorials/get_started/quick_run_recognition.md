### 1. Install PaddlePaddle and Download PLSC
See [installation](./installation.md)

### 2. Download Dummy Dataset and Unzip
``` bash
# download dummy dataset
wget https://plsc.bj.bcebos.com/dataset/MS1M_v3_One_Sample.tgz
# unzip
mkdir -p ./dataset/
tar -xzf MS1M_v3_One_Sample.tgz -C ./dataset/
# convert test bin file to images and label.txt
python plsc/data/dataset/tools/lfw_style_bin_dataset_converter.py --bin_path ./dataset/MS1M_v3_One_Sample/agedb_30.bin --out_dir ./dataset/MS1M_v3_One_Sample/agedb_30/ --flip_test
```

### 3. Run Train Scripts

#### 3.1 Single Node, 1 GPU:
``` bash
# Here, for simplicity, we just reuse the single node 8 gpus yaml configuration file.
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c ./plsc/configs/FaceRecognition/IResNet50_MS1MV3OneSample_ArcFace_0.1_1n8c_dp_fp32.yaml
```

#### 3.2 Single Node, 8 GPUs:
``` bash
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --nnodes=$PADDLE_NNODES --master=$PADDLE_MASTER --devices=$CUDA_VISIBLE_DEVICES tools/train.py -c ./plsc/configs/FaceRecognition/IResNet50_MS1MV3OneSample_ArcFace_0.1_1n8c_dp_fp32.yaml
```

### 4. Export Inference Model

#### 4.1 Single Node, 1 GPU:
``` bash
export CUDA_VISIBLE_DEVICES=0
python tools/export.py \
    -c ./plsc/configs/FaceRecognition/IResNet50_MS1MV3OneSample_ArcFace_0.1_1n8c_dp_fp32.yaml \
    -o Global.pretrained_model=output/IResNet50/latest \
    -o Model.data_format=NCHW
```

#### 4.2 Single Node, 8 GPUs:

``` bash
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    tools/export.py \
    -c ./plsc/configs/FaceRecognition/IResNet50_MS1MV3OneSample_ArcFace_0.1_1n8c_dp_fp32.yaml \
    -o Global.pretrained_model=output/IResNet50/latest \
    -o Model.data_format=NCHW
```
