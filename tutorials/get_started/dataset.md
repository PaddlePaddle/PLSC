## Recognition

### Download Dataset

Download the dataset from [insightface datasets](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_).

* MS1M_v2: MS1M-ArcFace
* MS1M_v3: MS1M-RetinaFace

### Extract MXNet Dataset to Images

```shell
# install mxnet firstly
pip install mxnet-cu112
python plsc/data/dataset/tools/mx_recordio_2_images.py --root_dir ms1m-retinaface-t1/ --output_dir dataset/MS1M_v3/
```

After finishing unzipping the dataset, the folder structure is as follows.

```
MS1M_v3
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

### Convert Test Dataset bin File to images and label.txt
```shell
python plsc/data/dataset/tools/lfw_style_bin_dataset_converter.py --bin_path ./dataset/MS1M_v3/agedb_30.bin --out_dir ./dataset/MS1M_v3/agedb_30/ --flip_test
```
