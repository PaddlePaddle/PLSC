# Face Recognition

The face recognition task is a case of achieving large-scale classification on PLSC, 
and the goal is to implement and reproduce the SOTA algorithm. It has 
the ability to train tens of millions of identities with high throughput in a single server.

Function has supported:
* ArcFace
* CosFace
* PartialFC
* SparseMomentum
* FP16 training
* DataParallel(backbone layer) + ModelParallel(FC layer) distributed training

Backbone includes:
* IResNet
* FaceViT

## Requirements
To enjoy some new features, PaddlePaddle 2.4 is required. For more installation tutorials 
refer to [installation.md](../../../tutorials/get_started/installation.md)

## Data Preparation

### Download Dataset

Download the dataset from insightface datasets.

- [MS1MV2](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-arcface-85k-ids58m-images-57) (87k IDs, 5.8M images)
- [MS1MV3](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_#ms1m-retinaface) (93k IDs, 5.2M images)
- [Glint360K](https://github.com/deepinsight/insightface/tree/master/recognition/partial_fc#4-download) (360k IDs, 17.1M images)
- [WebFace42M](https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/docs/prepare_webface42m.md) (2M IDs, 42.5M images)

Note:
* MS1MV2: MS1M-ArcFace
* MS1MV3: MS1M-RetinaFace
* WebFace42M: cleared WebFace260M

### [Optional] Extract MXNet Dataset to Images
```shell
# for example, here extract MS1MV3 dataset
python -m plsc.data.dataset.tools.mx_recordio_2_images --root_dir /path/to/ms1m-retinaface-t1/ --output_dir ./dataset/MS1M_v3/
```

### Extract LFW Style bin Dataset to Images
```shell
# for example, here extract agedb_30 bin to images
python -m plsc.data.dataset.tools.lfw_style_bin_dataset_converter --bin_path ./dataset/MS1M_v3/agedb_30.bin --out_dir ./dataset/MS1M_v3/agedb_30 --flip_test
```

### Dataset Directory
We put all the data in the `./dataset/` directory, and we also recommend using soft links, for example:
```shell
mkdir -p ./dataset/
ln -s /path/to/MS1M_v3 ./dataset/MS1M_v3
```

## How to Train

```bash
# Note: If running on multiple nodes, 
# set the following environment variables 
# and then need to run the script on each node.
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    plsc-train \
    -c ./configs/IResNet50_MS1MV3_ArcFace_pfc01_1n8c_dp_mp_fp16o1.yaml
```

## How to Export

```bash
# In general, we only need to export the 
# backbone, so we only need to run the 
# export command on a single device.
export PADDLE_NNODES=1
export PADDLE_MASTER="127.0.0.1:12538"
export CUDA_VISIBLE_DEVICES=0
python -m paddle.distributed.launch \
    --nnodes=$PADDLE_NNODES \
    --master=$PADDLE_MASTER \
    --devices=$CUDA_VISIBLE_DEVICES \
    plsc-export \
    -c ./configs/IResNet50_MS1MV3_ArcFace_pfc01_1n8c_dp_mp_fp16o1.yaml \
    -o Global.pretrained_model=./output/IResNet50/latest \
    -o FP16.level=O0 \ # export FP32 model when training with FP16
    -o Model.data_format=NCHW # IResNet required if training with NHWC 
```

## Evaluation IJB-C
```bash
python onnx_ijbc.py \
  --model-root ./output/IResNet50.onnx \
  --image-path ./ijb/IJBC/ \
  --target IJBC
```

## Model Zoo

|  Datasets  | Backbone                | Config                                                       | Devices   | PFC  | IJB-C(1E-4) | IJB-C(1E-5) | checkpoint                                                   | log                                                          |
| :--------: | :---------------------- | ------------------------------------------------------------ | --------- | ---- | ----------- | :---------- | :----------------------------------------------------------- | ------------------------------------------------------------ |
|   MS1MV3   | Res50                   | [config](./configs/IResNet50_MS1MV3_ArcFace_pfc10_1n8c_dp_mp_fp16o1.yaml) | N1C8*A100 | 1.0  | 96.52       | 94.60       | [download](https://plsc.bj.bcebos.com/models/face/v2.4/IResNet50_MS1MV3_ArcFace_pfc10_1n8c_dp_mp_fp16o1.pdparams) | [download](https://plsc.bj.bcebos.com/models/face/v2.4/IResNet50_MS1MV3_ArcFace_pfc10_1n8c_dp_mp_fp16o1.log) |
| WebFace42M | FaceViT_tiny_patch9_112 | [config](./configs/FaceViT_tiny_patch9_112_WebFace42M_CosFace_pfc10_droppath005_mask0_1n8c_dp_mp_fp16o1.yaml) | N1C8*A100 | 1.0  | 97.24       | 95.79       | [download](https://plsc.bj.bcebos.com/models/face/v2.4/FaceViT_tiny_patch9_112_WebFace42M_CosFace_pfc10_droppath005_mask0_1n8c_dp_mp_fp16o1.pdparams) | [download](https://plsc.bj.bcebos.com/models/face/v2.4/FaceViT_tiny_patch9_112_WebFace42M_CosFace_pfc10_droppath005_mask0_1n8c_dp_mp_fp16o1.log) |
| WebFace42M | FaceViT_tiny_patch9_112 | [config](./configs/FaceViT_tiny_patch9_112_WebFace42M_CosFace_pfc02_droppath005_mask0_1n8c_dp_mp_fp16o1.yaml) | N1C8*A100 | 0.2  | 97.28       | 95.79       | [download](https://plsc.bj.bcebos.com/models/face/v2.4/FaceViT_tiny_patch9_112_WebFace42M_CosFace_pfc02_droppath005_mask0_1n8c_dp_mp_fp16o1.pdparams) | [download](https://plsc.bj.bcebos.com/models/face/v2.4/FaceViT_tiny_patch9_112_WebFace42M_CosFace_pfc02_droppath005_mask0_1n8c_dp_mp_fp16o1.log) |
| WebFace42M | FaceViT_base_patch9_112 | [config](./configs/FaceViT_base_patch9_112_WebFace42M_CosFace_pfc03_droppath005_mask005_1n8c_dp_mp_fp16o1.yaml) | N1C8*A100 | 0.3  | 97.97       | 97.04       | [download](https://plsc.bj.bcebos.com/models/face/v2.4/FaceViT_base_patch9_112_WebFace42M_CosFace_pfc03_droppath005_mask005_1n8c_dp_mp_fp16o1.pdparams) | [download](https://plsc.bj.bcebos.com/models/face/v2.4/FaceViT_base_patch9_112_WebFace42M_CosFace_pfc03_droppath005_mask005_1n8c_dp_mp_fp16o1.log) |

## Citations

```
@misc{plsc,
    title={PLSC: An Easy-to-use and High-Performance Large Scale Classification Tool},
    author={PLSC Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PLSC}},
    year={2022}
}
@inproceedings{deng2019arcface,
  title={Arcface: Additive angular margin loss for deep face recognition},
  author={Deng, Jiankang and Guo, Jia and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={4690--4699},
  year={2019}
}
@inproceedings{An_2022_CVPR,
    author={An, Xiang and Deng, Jiankang and Guo, Jia and Feng, Ziyong and Zhu, XuHan and Yang, Jing and Liu, Tongliang},
    title={Killing Two Birds With One Stone: Efficient and Robust Training of Face Recognition CNNs by Partial FC},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month={June},
    year={2022},
    pages={4042-4051}
}
```
