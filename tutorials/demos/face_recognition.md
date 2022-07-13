### 1. Install PaddlePaddle and Download PLSC
See [installation](../get_started/installation.md)

### 2. Download Dataset and Prepare
See [dataset](../get_started/dataset.md)

### 3. Run Train Scripts

#### 3.1 Backbone DP8 + PartialFC MP8, MS1MV3
```bash
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --nnodes=$PADDLE_NNODES --master=$PADDLE_MASTER --devices=$CUDA_VISIBLE_DEVICES tools/train.py -c ./plsc/configs/FaceRecognition/IResNet50_MS1MV3_ArcFace_0.1_1n8c_dp_mp_fp16o2.yaml
```

#### 3.2 Backbone DP16 + PartialFC MP16, WebFace42M

Note that setting ``PADDLE_MASTER="xxx.xxx.xxx.xxx:port"`` according your network environment.
##### Node0
```bash
export PADDLE_NNODES=2
export PADDLE_MASTER="192.168.210.10:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --nnodes=$PADDLE_NNODES --master=$PADDLE_MASTER --devices=$CUDA_VISIBLE_DEVICES tools/train.py -c ./plsc/configs/FaceRecognition/IResNet50_WebFace42M_CosFace_0.2_2n16c_dp_mp_fp16o2.yaml
```

##### Node1
```bash
export PADDLE_NNODES=2
export PADDLE_MASTER="192.168.210.10:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --nnodes=$PADDLE_NNODES --master=$PADDLE_MASTER --devices=$CUDA_VISIBLE_DEVICES tools/train.py -c ./plsc/configs/FaceRecognition/IResNet50_WebFace42M_CosFace_0.2_2n16c_dp_mp_fp16o2.yaml
```

#### 3.3 Backbone DP8 + PartialFC DP8, MS1MV3
```bash
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m paddle.distributed.launch --nnodes=$PADDLE_NNODES --master=$PADDLE_MASTER --devices=$CUDA_VISIBLE_DEVICES tools/train.py -c ./plsc/configs/FaceRecognition/IResNet50_MS1MV3_ArcFace_0.1_1n8c_dp_fp16o2.yaml
```


### 4. Export ONNX model

The way to export an ONNX model is simple, the interface is ``tools/export.py``. In order to successfully load the pretrained checkpoint, we also need to configure the same yaml file. The additional difference is that some settings in the yaml configuration file can be overridden via the ``-o`` parameter. 

For example, ``-o Global.pretrained_model=output/IResNet50/latest`` can configure the checkpoint directory, and ``-o Model.data_format=NCHW`` can modify the data layout.

Note that, we support export to ONNX model and Paddle Inference Model. You can set ``-o Export.export_type=paddle`` to export Paddle Inference Model or ``-o Export.export_type=onnx`` to export ONNX Model.

Backbone DP8 + PartialFC MP8, MS1MV3 can export the ONNX model from the script below, and other cases can be configured according to the actual situation.

``` bash
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    tools/export.py \
    -c ./plsc/configs/FaceRecognition/IResNet50_MS1MV3_ArcFace_0.1_1n8c_dp_mp_fp16o2.yaml \
    -o Global.pretrained_model=output/IResNet50/latest \
    -o Model.data_format=NCHW
```

### 5. Evaluation on the IJBC dataset
Coming soon...
