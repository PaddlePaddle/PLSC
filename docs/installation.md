# 安装说明

## 1. 安装PaddlePaddle

版本要求：

* PaddlePaddle >= 1.6.2
* Python 2.7+

关于PaddlePaddle对操作系统、CUDA、cuDNN等软件版本的兼容信息，请查看[PaddlePaddle安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/install/index_cn.html)


### pip安装

当前，需要在GPU版本的PaddlePaddle下使用大规模分类库。

```shell
pip install paddlepaddle-gpu
```

### Conda安装

PaddlePaddle支持Conda安装，减少相关依赖模块的安装成本。conda相关使用说明可以参考[Anaconda](https://www.anaconda.com/distribution/)

```shell
conda install -c paddle paddlepaddle-gpu cudatoolkit=9.0
```

* 请安装NVIDIA NCCL >= 2.4.7，并在Linux系统下运行。

更多安装方式和信息请参考[PaddlePaddle安装说明](https://www.paddlepaddle.org.cn/documentation/docs/zh/beginners_guide/install/index_cn.html)

## 2. 安装依赖包

```shell
pip install -r requirements.txt
```

## 3. 安装大规模分类库

```shell
pip install plsc
```
