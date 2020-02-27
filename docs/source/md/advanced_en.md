# Advanced Usage

## Checkpoints Uploading and downloading for HDFS
With PLSC, checkpoints can be uploaded to or downloaded from HDFS file systems automatically when HDFS information is set by *set_hdfs_info* api.

### Checkpoints Uploading
The following example code shows how to upload checkpoints to HDFS:
```python
from plsc import Entry

if __name__ == "__main__":
    ins = Entry()
    ins.set_model_save_dir('./saved_model')
    ins.set_hdfs_info("your_hdfs_addr", "name,passwd", "some_dir_on_hdfs")
    ins.train()
```

### Checkpoints Downloading
The following code snippet shows how to download checkpoints from HDFS:
```python
from plsc import Entry

if __name__ == "__main__":
    ins = Entry()
    ins.set_checkpoint_dir('./saved_model')
    ins.set_hdfs_info("your_hdfs_addr", 
                      "name,passwd",
                      fs_checkpoint_dir="some_dir")
```

Using above code, checkpoints in "some_dir" on HDFS file systems will be downloaded on local "./saved_model" directory. Please make sure the local directory "./saved_model" exists.

## Pre-processing for images in base64 format
In practice, base64 is a common format to store images. All image data is stored in one file and each line of the file represents a image data in base64 format and its corresponding label.

The following gives an example structure of datasets:
```shell script
dataset
     |-- file_list.txt
     |-- dataset.part1
     |-- dataset.part2
     |   ....
     `-- dataset.part10
```

The file file_list.txt records all data files, each line of which represents a data file, for example dataset.part1.

For distributed training, every GPU card has to process the same number of data, and global shuffle is usually used for all images.

The provided pre-processing tool does global shuffle on all training images and splits these images into groups evenly. The number of groups is equal to the number of GPU cards used.

### How to use
The pre-processing tool is put in the directory "tools". To use it, the sqlite3 module is required, which can be installed using the following command:
```shell script
pip install sqlite3
``` 

Using the following command to show the help message:
```shell script
python tools/process_base64_files.py --help
```

The tool provides the following options:
- data_dir: the root directory for datasets
- file_list: the file to record data files, e.g., file_list.txt
- nranks: number of final data files

The usage of the tool is as follows:
```shell script
python tools/process_base64_files.py --data_dir=./dataset --file_list=file_list.txt --nranks=8
```

Then, eight data files will be generated, each of which contains the same number of samples.

Note: plsc.utils.base64_reader can be used to reader images stored in the format of base64.

## Mixed Precision Training
PLSC supports mixed precision training, which can be used to improve training speed and decrease memory usage.

### How to use
Using the following code to enable mixed precision training:
```python
from plsc import Entry

def main():
    ins = Entry()
    ins.set_mixed_precision(True)
    ins.train()

if __name__ == "__main__":
    main()
```

For more information about mixed precision training please refer to:
- Paper: [MIXED PRECISION TRAINING](https://arxiv.org/abs/1710.03740)
- Nvidia Guider: [Training With Mixed Precision](https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html)

## User-defined Models
By default, PLSC will use ResNet50 model, and it provides ResNet101 and ResNet152 models as well.

Users can define their own models based on the base class plsc.models.base_model.BaseModel by implementing the *build_network* method.

The following code shows how to define a custom model:
```python
import paddle.fluid as fluid
from plsc import Entry
from plsc.models.base_model import BaseModel

class ResNet(BaseModel):
    def __init__(self, layers=50, emb_dim=512):
        super(ResNet, self).__init__()
        self.layers = layers
        self.emb_dim = emb_dim

    def build_network(self,
                      input,
                      label,
                      is_train):
        layers = self.layers
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers {}, but given {}".format(supported_layers, layers)

        if layers == 50:
            depth = [3, 4, 14, 3]
            num_filters = [64, 128, 256, 512]
        elif layers == 101:
            depth = [3, 4, 23, 3]
            num_filters = [256, 512, 1024, 2048]
        elif layers == 152:
            depth = [3, 8, 36, 3]
            num_filters = [256, 512, 1024, 2048]

        conv = self.conv_bn_layer(input=input,
                                  num_filters=64,
                                  filter_size=3,
                                  stride=1,
                                  pad=1,
                                  act='prelu',
                                  is_train=is_train)

        for block in range(len(depth)):
            for i in range(depth[block]):
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=num_filters[block],
                    stride=2 if i == 0 else 1,
                    is_train=is_train)

        bn = fluid.layers.batch_norm(input=conv,
                                     act=None,
                                     epsilon=2e-05,
                                     is_test=False if is_train else True)
        drop = fluid.layers.dropout(
            x=bn,
            dropout_prob=0.4,
            dropout_implementation='upscale_in_train',
            is_test=False if is_train else True)
        fc = fluid.layers.fc(
            input=drop,
            size=self.emb_dim,
            act=None,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Xavier(uniform=False, fan_in=0.0)),
            bias_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer()))
        emb = fluid.layers.batch_norm(input=fc,
                                      act=None,
                                      epsilon=2e-05,
                                      is_test=False if is_train else True)
        return emb

	def conv_bn_layer(
        ... ...

if __name__ == "__main__":
    ins = Entry()
    ins.set_model(ResNet())
    ins.train()
```

## How to use custom training data
With PLSC, we assume the dataset is organized in the following structure:
```shell script
train_data/
|-- images
`-- label.txt
```
All images are stored in the directory 'images', and the file 'label.txt' are used to record the index of all images, each line of which represents the relative path for a image and its corresponding label.

When dataset for users are organized in their own structure, using the following steps to use their own datasets:
1. Define a generator, which pre-processing users' images (e.g., resizing) and generate samples one by one using *yield*;
    *  A sample is a tuple of (data, label), where data represents a images after decoding and preprocessing
2. Use paddle.batch to wrap the above generator, and get the batched generator
3. Assign the batched reader to the 'train_reader' member of plsc.Entry

We assume the dataset for a user is organized as follows:
```shell script
train_data/
|-- images
`-- label.txt
```
First, using the following code to define a generator:
```python
import random
import os
from PIL import Image

def arc_train(data_dir):
    label_file = os.path.join(data_dir, 'label.txt')
    train_image_list = None
    with open(label_file, 'r') as f:
        train_image_list = f.readlines()
    train_image_list = get_train_image_list(data_dir)

    def reader():
        for j in range(len(train_image_list)):
            path, label = train_image_list[j]
            path = os.path.join(data_dir, path)
            img = Image.open(path)
            if random.randint(0, 1) == 1:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = np.array(img).astype('float32').transpose((2, 0, 1))
            yield img, label

    return reader
```

The following example code shows how to use the custom dataset:
```python
import argparse
import paddle
from plsc import Entry
import reader

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",
                    type=str,
                    default="./data",
                    help="Directory for datasets.")
args = parser.parse_args()


def main():
    global args
    ins = Entry()
    ins.set_dataset_dir(args.data_dir)
    train_reader = reader.arc_train(args.data_dir)
    # Batch the above samples;
    batched_train_reader = paddle.batch(train_reader,
                                        ins.train_batch_size)
    # Set the reader to use during training to the above batch reader.
    ins.train_reader = batched_train_reader

    ins.train()


if __name__ == "__main__":
    main()
```

For more examples, please refer to [example](../../../demo/custom_reader.py).