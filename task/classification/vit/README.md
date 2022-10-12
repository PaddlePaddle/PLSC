# Vision Transformer

## Introduction
PaddlePaddle reimplementation of [Google's repository for the ViT model](https://github.com/google-research/vision_transformer) that was released with the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) by Alexey Dosovitskiy\*†, Lucas Beyer\*, Alexander Kolesnikov\*, Dirk
Weissenborn\*, Xiaohua Zhai\*, Thomas Unterthiner, Mostafa Dehghani, Matthias
Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit and Neil Houlsby\*†.

(\*) equal technical contribution, (†) equal advising.

![Figure 1 from paper](https://github.com/google-research/vision_transformer/raw/main/vit_figure.png)

Overview of the model: we split an image into fixed-size patches, linearly embed
each of them, add position embeddings, and feed the resulting sequence of
vectors to a standard Transformer encoder. In order to perform classification,
we use the standard approach of adding an extra learnable "classification token"
to the sequence.

## Installation
- See [installation.md](../../../tutorials/get_started/installation.md)

Note: All commands are executed in the PLSC root directory.

```bash
cd /path/to/PLSC
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
    tools/train.py \
    -c ./plsc/configs/VisionTransformer/ViT_base_patch16_224_in1k_1n8c_dp_fp16o2.yaml
```

## How to Finetune

```bash
# [Optional] Download checkpoint
mkdir -p pretrained/vit/ViT_base_patch16_224/
wget -O ./pretrained/vit/imagenet2012-ViT-B_16-224.pdparams https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet2012-ViT-B_16-224.pdparams

```


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
    tools/train.py \
    -c ./plsc/configs/VisionTransformer/ViT_base_patch16_384_ft_in1k_1n8c_dp_fp16o2.yaml \
    -o Global.pretrained_model=./pretrained/vit/ViT_base_patch16_224/imagenet2012-ViT-B_16-224.pdparams \
```

## Other Configurations
We provide more directly runnable configurations, see [ViT Configurations](../../../plsc/configs/VisionTransformer).


## Models

| Model        | Phase    | Dataset      | GPUs      | Img/sec | Top1 Acc | Pre-trained checkpoint                                       | Fine-tuned checkpoint                                        | Log                                                          |
| ------------ | -------- | ------------ | --------- | ------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ViT-B_16_224 | pretrain | ImageNet2012 | A100*N1C8 | 3583    | 0.75196  | [download](https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet2012-ViT-B_16-224.pdparams) | -                                                            | [log](https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet2012-ViT-B_16-224.log) |
| ViT-B_16_224 | finetune | ImageNet2012 | A100*N1C8 | 719     | 0.77972  | [download](https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet2012-ViT-B_16-224.pdparams) | [download](https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet2012-ViT-B_16-384.pdparams) | [log](https://plsc.bj.bcebos.com/models/vit/v2.4/imagenet2012-ViT-B_16-384.log) |


## Citations

```bibtex
@article{dosovitskiy2020,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and Beyer, Lucas and Kolesnikov, Alexander and Weissenborn, Dirk and Zhai, Xiaohua and Unterthiner, Thomas and  Dehghani, Mostafa and Minderer, Matthias and Heigold, Georg and Gelly, Sylvain and Uszkoreit, Jakob and Houlsby, Neil},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```
