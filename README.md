<p align="center">
  <img src="./plsc-logo.png?raw=true" align="middle"  width="500" />
</p>

------------------------------------------------------------------------------------------

<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/PLSC/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/PLSC?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href="https://github.com/PaddlePaddle/PLSC/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/PLSC?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/PLSC/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/PLSC?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/PLSC/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/PLSC?color=ccf"></a>
</p>

## Introduction

[PLSC](https://github.com/PaddlePaddle/PLSC) is an open source repo for a collection of Paddle Large Scale Classification Tools, which supports large-scale classification model pre-training as well as finetune for downstream tasks.

## Available Models
* [Face Recognition](./task/recognition/face/)
* [ViT](./task/classification/vit/)
* [Swin](./task/classification/swin/)
* [DeiT](./task/classification/deit/)
* [CaiT](./task/classification/cait/)
* [ConvNeXt](./task/classification/convnext)
* [MoCo v3](./task/ssl/mocov3/)
* [CAE](./task/ssl/cae/)
* [MAE](./task/ssl/mae/)
* [ConvMAE](./task/ssl/mae/)
* [ToMe](./task/accelerate/tome)

## Top News 🔥

**Update (2023-01-11):** PLSC v2.4 is released, we refactored the entire repository based on task types. This repository has been adapted to PaddlePaddle release 2.4. In terms of models, we have added 4 new ones, including [FaceViT](https://arxiv.org/abs/2203.15565), [CaiT](https://arxiv.org/abs/2103.17239), [MoCo v3](https://arxiv.org/abs/2104.02057), [MAE](https://arxiv.org/abs/2111.06377). At present, each model in the repository can be trained from scratch to achieve the original official accuracy, especially the training of ViT-Large on the ImageNet21K dataset. In addition, we also provide a method for ImageNet21K data preprocessing. In terms of AMP training, PLSC uses FP16 O2 training by default, which can speed up training while maintaining accuracy.

**Update (2022-07-18):** PLSC v2.3 is released, a new upgrade, more modular and highly extensible. Support more tasks, such as [ViT](https://arxiv.org/abs/2010.11929), [DeiT](https://arxiv.org/abs/2012.12877). The `static` graph mode will no longer be maintained as of this release.

**Update (2022-01-11):** Supported NHWC data format of FP16 to improve 10% throughtput and decreased 30% GPU memory. It supported 92 million classes on single node 8 NVIDIA V100 (32G) and has high training throughtput. Supported best checkpoint save. And we released 18 pretrained models and PLSC v2.2.

**Update (2021-12-11):** Released [Zhihu Technical Artical](https://zhuanlan.zhihu.com/p/443091282) and [Bilibili Open Class](https://www.bilibili.com/video/BV1VP4y1G73X)

**Update (2021-10-10):** Added FP16 training, improved throughtput and optimized GPU memory. It supported 60 million classes on single node 8 NVIDIA V100 (32G) and has high training throughtput.

**Update (2021-09-10):** This repository supported both `static` mode and `dynamic` mode to use paddlepaddle v2.2, which supported 48 million classes on single node 8 NVIDIA V100 (32G). It added [PartialFC](https://arxiv.org/abs/2010.05222), SparseMomentum, and [ArcFace](https://arxiv.org/abs/1801.07698), [CosFace](https://arxiv.org/abs/1801.09414) (we refer to MarginLoss). Backbone includes IResNet and MobileNet.


## Installation

PLSC provides two usage methods: one is as an external third-party library, and users can use `import plsc` in their own projects; the other is to develop and use it locally based on this repository.

**Note**: With the continuous iteration of the PaddlePaddle version, the PLSC `master` branch adapts to the PaddlePaddle `develop` branch, and API mismatches may occur in lower versions of PaddlePaddle.

### Install plsc as a third-party library

```shell
pip install git+https://github.com/PaddlePaddle/PLSC@master
```

For stable development, you can install a previous version of plsc.

```
pip install plsc==2.4
```

### Install plsc locally

```shell
git clone https://github.com/PaddlePaddle/PLSC.git
cd /path/to/PLSC/
# [optional] pip install -r requirements.txt
python setup.py develop
```

See [Installation instructions](./tutorials/get_started/installation.md).


## Getting Started

See [Quick Run Recognition](./tutorials/get_started/quick_run_recognition.md) for the basic usage of PLSC.

## Tutorials

* [Configuration](./tutorials/basic/config.md)

See more [tutorials](./tutorials/README.md).

## Documentation

* [Data Augmentation](./docs/data_augmentation.md)

See [documentation](./docs/README.md) for the usage of more APIs or modules.


## License

This project is released under the [Apache 2.0 license](./LICENSE).

## Citation

```
@misc{plsc,
    title={PLSC: An Easy-to-use and High-Performance Large Scale Classification Tool},
    author={PLSC Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PLSC}},
    year={2022}
}
```
