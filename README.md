# Arcface-Paddle

## 1. Introduction

`Arcface-Paddle` is an open source deep face detection and recognition toolkit, powered by PaddlePaddle. `Arcface-Paddle` provides three related pretrained models now, include `BlazeFace` for face detection, `ArcFace` and `MobileFace` for face recognition.

- This tutorial is mainly about face recognition.
- For face detection task, please refer to: [Face detection tuturial](../../detection/blazeface_paddle/README_en.md).

## 2. Environment preparation

Please refer to [Installation](./install_en.md) to setup environment at first.


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

### 8.1 Performance on IJB-C and Verification Datasets

**Configuration：**
  * GPU: 8 NVIDIA Tesla V100 32G
  * Precison: AMP
  * BatchSize: 128/1024

| Mode    | Datasets | backbone | Ratio | IJBC(1e-05) | IJBC(1e-04) | agedb30 | cfp_fp | lfw  | log  |
| ------- | :------: | :------- | ----- | :---------- | :---------- | :------ | :----- | :--- | :--- |
| Static  |  MS1MV3  | r50      | 0.1   |             |             |         |        |      |      |
| Static  |  MS1MV3  | r50      | 1.0   |             |             |         |        |      |      |
| Dynamic |  MS1MV3  | r50      | 0.1   |             |             |         |        |      |      |
| Dynamic |  MS1MV3  | r50      | 1.0   |             |             |         |        |      |      |

  
### 8.2 Maximum Number of Identities 

**Configuration：**
  * GPU: 8 NVIDIA Tesla V100 32G
  * Precison: AMP
  * BatchSize: 64/512
  * SampleRatio: 0.1

| Mode                      | Res50                        | Res100                       |
| ------------------------- | ---------------------------- | ---------------------------- |
| Oneflow                   |                              |                              |
| PyTorch                   |                              |                              |
| Paddle (static)           |                              |                              |
| Paddle (dynamic)          |                              |                              |


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
