### 1. Install PaddlePaddle and Download PLSC
See [installation](./installation.md)

### 2. Data Preparation 
#### 2.1 Download Dummy Dataset and Unzip
``` bash
# download dummy dataset
wget https://plsc.bj.bcebos.com/dataset/MS1M_v3_One_Sample.tgz
# unzip
mkdir -p ./dataset/
tar -xzf MS1M_v3_One_Sample.tgz -C ./dataset/
```
#### 2.2 Extract LFW Style bin Dataset to Images
```bash
python plsc/data/dataset/tools/lfw_style_bin_dataset_converter.py --bin_path ./dataset/MS1M_v3_One_Sample/agedb_30.bin --out_dir ./dataset/MS1M_v3_One_Sample/agedb_30/ --flip_test
```

### 3. Run Train Scripts

#### 3.1 Single Node with 1 GPU

Run the script from command line.
``` bash
# Here, for simplicity, we just reuse the single node 8 gpus yaml configuration file.
export CUDA_VISIBLE_DEVICES=0
plsc-train \
-c task/recognition/face/configs/IResNet50_MS1MV3_ArcFace_pfc10_1n8c_dp_mp_fp16o1.yaml \
-o DataLoader.Train.dataset.image_root=./dataset/MS1M_v3_One_Sample \
-o DataLoader.Train.dataset.cls_label_path=./dataset/MS1M_v3_One_Sample/label.txt \
-o DataLoader.Eval.dataset.image_root=./dataset/MS1M_v3_One_Sample/agedb_30 \
-o DataLoader.Eval.dataset.cls_label_path=./dataset/MS1M_v3_One_Sample/agedb_30/label.txt```
```

#### 3.2 Single Node with 8 GPUs 

Run the script from command line.
``` bash
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES 
    plsc-train \
    -c task/recognition/face/configs/IResNet50_MS1MV3_ArcFace_pfc10_1n8c_dp_mp_fp16o1.yaml \
    -o DataLoader.Train.dataset.image_root=./dataset/MS1M_v3_One_Sample \
    -o DataLoader.Train.dataset.cls_label_path=./dataset/MS1M_v3_One_Sample/label.txt \
    -o DataLoader.Eval.dataset.image_root=./dataset/MS1M_v3_One_Sample/agedb_30 \
    -o DataLoader.Eval.dataset.cls_label_path=./dataset/MS1M_v3_One_Sample/agedb_30/label.txt
```
You can also run the shell file directly,
```bash
sh tools/train.sh
```

### 4. Export Inference Model

#### 4.1 Single Node, 1 GPU:
``` bash
export CUDA_VISIBLE_DEVICES=0
plsc-export \
-c task/recognition/face/configs/IResNet50_MS1MV3_ArcFace_pfc10_1n8c_dp_mp_fp16o1.yaml \
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
    plsc-export \
    -c task/recognition/face/configs/IResNet50_MS1MV3_ArcFace_pfc10_1n8c_dp_mp_fp16o1.yaml \
    -o Global.pretrained_model=output/IResNet50/latest \
    -o Model.data_format=NCHW
```
You can also run the shell file directly,
```bash
sh tools/export.sh
```
