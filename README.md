# PLSC

## 1. Introduction

`PLSC` is an open source Paddle Large Scale Classification Tools, which supports 60 million classes on single NVIDIA V100 (32G).

## 2. Environment preparation

Please refer to [Installation](./install.md) to setup environment at first.


## 3. Data preparation

### 3.1 Enter recognition dir.

```
cd /path/to/arcface_paddle/
```

### 3.2 Download

Download the dataset from [https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_).

### 3.3 Extract MXNet Dataset to images

```shell
python tools/mx_recordio_2_images.py --root_dir ms1m-retinaface-t1/ --output_dir MS1M_v3/
```

After finishing unzipping the dataset, the folder structure is as follows.

```
arcface_paddle/MS1M_v3
|_ images
|  |_ 00000001.jpg
|  |_ ...
|  |_ 05179510.jpg
|_ label.txt
|_ agedb_30.bin
|_ cfp_ff.bin
|_ cfp_fp.bin
|_ lfw.bin
```

Label file format is as follows.

```
# delimiter: "\t"
# the following the content of label.txt
images/00000001.jpg 0
...
```

If you want to use customed dataset, you can arrange your data according to the above format. 

### 3.3 Transform between original image files and bin files

If you want to convert original image files to `bin` files used directly for training process, you can use the following command to finish the conversion.

```shell
python tools/convert_image_bin.py --image_path="your/input/image/path" --bin_path="your/output/bin/path" --mode="image2bin"
```

If you want to convert `bin` files to original image files, you can use the following command to finish the conversion.

```shell
python tools/convert_image_bin.py --image_path="your/input/bin/path" --bin_path="your/output/image/path" --mode="bin2image"
```

## 4. How to Training

### 4.1 Single node, 8 GPUs:

#### Static Mode

```bash
sh scripts/train_static.sh
```

#### Dynamic Mode

```bash
sh scripts/train_dynamic.sh
```


During training, you can view loss changes in real time through `VisualDL`,  For more information, please refer to [VisualDL](https://github.com/PaddlePaddle/VisualDL/).


## 5. Model evaluation

The model evaluation process can be started as follows.

#### Static Mode

```bash
sh scripts/validation_static.sh
```

#### Dynamic Mode

```bash
sh scripts/validation_dynamic.sh
```

## 6. Export model
PaddlePaddle supports inference using prediction engines. Firstly, you should export inference model.

#### Static Mode

```bash
sh scripts/export_static.sh
```

#### Dynamic Mode

```bash
sh scripts/export_dynamic.sh
```

We also support export to onnx model, you only need to set `--export_type onnx`.

## 7. Model inference

The model inference process supports paddle save inference model and onnx model.

```bash
sh scripts/inference.sh
```

## 8. Model performance

### 8.1 Performance on Verification Datasets

**Configuration：**
  * GPU: 8 NVIDIA Tesla V100 32G
  * Precison: Pure FP16
  * BatchSize: 128/1024

| Mode    | Datasets | backbone | Ratio | agedb30 | cfp_fp | lfw  | log  | checkpoint |
| ------- | :------: | :------- | ----- | :------ | :----- | :--- | :--- |  :--- |
| Static  |  MS1MV3  | r50      | 0.1   | 0.98317 | 0.98943| 0.99850 | [log](https://raw.githubusercontent.com/GuoxiaWang/plsc_log/master/static/ms1mv3_r50_static_128_fp16_0.1/training.log) | [checkpoint](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/distributed/ms1mv3_r50_static_128_fp16_0.1_epoch_24.tgz) |
| Static  |  MS1MV3  | r50      | 1.0   | 0.98283 | 0.98843| 0.99850 | [log](https://raw.githubusercontent.com/GuoxiaWang/plsc_log/master/static/ms1mv3_r50_static_128_fp16_1.0/training.log) | [checkpoint](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/distributed/ms1mv3_r50_static_128_fp16_1.0_epoch_24.tgz) |
| Dynamic |  MS1MV3  | r50      | 0.1   | 0.98333 | 0.98900| 0.99833 | [log](https://raw.githubusercontent.com/GuoxiaWang/plsc_log/master/dynamic/ms1mv3_r50_dynamic_128_fp16_0.1/training.log) | [checkpoint](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/distributed/ms1mv3_r50_dynamic_128_fp16_0.1_eopch_24.tgz) |
| Dynamic |  MS1MV3  | r50      | 1.0   | 0.98317 | 0.98900| 0.99833 | [log](https://raw.githubusercontent.com/GuoxiaWang/plsc_log/master/dynamic/ms1mv3_r50_dynamic_128_fp16_1.0/training.log) | [checkpoint](https://paddle-model-ecology.bj.bcebos.com/model/insight-face/distributed/ms1mv3_r50_dynamic_128_fp16_1.0_eopch_24.tgz) |

  
### 8.2 Maximum Number of Identities 

**Configuration：**
  * GPU: 8 NVIDIA Tesla V100 32G
  * Precison: Pure FP16
  * BatchSize: 64/512
  * SampleRatio: 0.1

| Mode                      | Res50                        | Res100                       |
| ------------------------- | ---------------------------- | ---------------------------- |
| Paddle (static)           | 60000000 (32018MiB/32510MiB) | 60000000 (32018MiB/32510MiB) |
| Paddle (dynamic)          | 59000000 (31970MiB/32510MiB) | 59000000 (31970MiB/32510MiB) |

**Note:** config environment variable ``export FLAGS_allocator_strategy=naive_best_fit``

### 8.3 Throughtput

**Configuration：**
  * BatchSize: 128/1024
  * SampleRatio: 0.1
  * Datasets: MS1MV3
  
![insightface_throughtput](https://github.com/GuoxiaWang/plsc_log/blob/master/insightface_throughtput.png)

## 9. Demo

Combined with face detection model, we can complete the face recognition process.

Firstly, use the following commands to download the index gallery, demo image and font file for visualization.


```bash
# Index library for the recognition process
wget https://raw.githubusercontent.com/littletomatodonkey/insight-face-paddle/main/demo/friends/index.bin
# Demo image
wget https://raw.githubusercontent.com/littletomatodonkey/insight-face-paddle/main/demo/friends/query/friends2.jpg
# Font file for visualization
wget https://raw.githubusercontent.com/littletomatodonkey/insight-face-paddle/main/SourceHanSansCN-Medium.otf
```

The demo image is shown as follows.

<div align="center">
<img src="https://raw.githubusercontent.com/littletomatodonkey/insight-face-paddle/main/demo/friends/query/friends2.jpg"  width = "800" />
</div>


Use the following command to run the whole face recognition demo.

```shell
# detection + recogniotion process
python tools/test_recognition.py --det --rec --index=index.bin --input=friends2.jpg --output="./output"
```

The final result is save in folder `output/`, which is shown as follows.

<div align="center">
<img src="https://raw.githubusercontent.com/littletomatodonkey/insight-face-paddle/main/demo/friends/output/friends2.jpg"  width = "800" />
</div>

For more details about parameter explanations, index gallery construction and whl package inference, please refer to [Whl package inference tutorial](https://github.com/littletomatodonkey/insight-face-paddle).
