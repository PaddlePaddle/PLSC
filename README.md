English | [Simplified Chinese](./README_zh.md)
# PLSC: PaddlePaddle Large Scale Classification Package

## Introduction
The fully-connected layer is usually used as the last layer of deep neural networks for multi-class classification tasks. As the memory occupied by parameters of the fully-connected layer increasing linearly with the increase in number of classes, the training of those kind of neural networks face the following challenges when tasks have a large number of classes:

* GPU memory limitation

* Heavy communication overhead

**P**addlePaddle **L**arge **S**cale **C**lassification (PLSC) package is an end-to-end solution for large scale classification tasks based on [PaddlePaddle](https://github.com/PaddlePaddle/Paddle), which provides a total solution from model training to deployment for users.

## Highlights
* Support large scale classification problems：PLSC can support 2.5x more classes using 8 V100 GPU cards, and the maximum number of classes supported increases as the increase in number of GPU cards used;
* High training speed：With one million classes, the training speed is 2,122 images/sec using ResNet50 model and 8 V100 GPU cards are used, and PLSC supports mixed precision training and distributed training;
* Support the adjustment of number of GPU cards used for training：A training starts from checkpoints can use a different number of GPU cards from that used for the last training;
* Builtin pre-precessing tools for base64 format images：PLSC provides builtin pre-processing tools for images in base64 format，including global shuffle, dataset spliting, and so on;
* Support for custom models：PLSC contains the builtin ResNet50, ResNet101 and ResNet152 models，and users can define their customized models；
* Support the automatic uploading and downloading of checkpoints for HDFS file systems;
* A end-to-end solution：PLSC provides a total solution from model training to deployment for users.

## Quick Start
Refer to [quick start](docs/source/md/quick_start_en.md) for how to install and use PLSC.

## Serving and Deployment
Refer to [deployment guider](docs/source/md/serving_en.md) for how to use and deployment serving service.

## Advanced Usage
Refer to [advanced guider](docs/source/md/advanced_en.md) for advanced usage.

## API Reference
Refer to [API reference](docs/source/md/api_reference_en.md) for all APIs.

## Pre-trained model and performance

### Pre-trained model

We provide the following pre-trained model for users to fine-tuning their models.

| Model            | Description          |
| :--------------- | :------------- |
| [resnet50_distarcface_ms1m_arcface](https://plsc.bj.bcebos.com/pretrained_model/resnet50_distarcface_ms1mv2.tar.gz) | This model is trained with ResNet50 and MS1M-ArcFace and the loss type used is dist_arcface. The validation accuracy on lfw dataset is 0.99817. | 

### Performance

| Model            | Training Dataset | lfw      | agendb_30 | cfp_ff   | cfp_fp |
| :--------------- | :------------- | :------ | :-----     | :------ | :----  |
| ResNet50         | MS1M-ArcFace   | 0.99817 | 0.99827    | 0.99857 | 0.96314 |
| ResNet50         | CASIA          | 0.9895  | 0.9095     | 0.99057 | 0.915 |

Note：The loss_type used is 'dist_arcface'. For more information about ArcFace please refer to  [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698).
