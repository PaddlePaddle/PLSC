# This demo shows how to use user-defined training dataset.
# The following steps are needed to use user-defined training datasets:
# 1. Build a reader, which preprocess images and yield a sample in the
#    format (data, label) each time, where data is the decoded image data;
# 2. Batch the above samples;
# 3. Set the reader to use during training to the above batch reader.

import argparse

import paddle
from plsc import Entry
from plsc.utils import jpeg_reader as reader

parser = argparse.ArgumentParser()
parser.add_argument("--model_save_dir",
                    type=str,
                    default="./saved_model",
                    help="Directory to save models.")
parser.add_argument("--data_dir",
                    type=str,
                    default="./data",
                    help="Directory for datasets.")
parser.add_argument("--num_epochs",
                    type=int,
                    default=2,
                    help="Number of epochs to run.")
parser.add_argument("--loss_type",
                    type=str,
                    default='arcface',
                    help="Loss type to use.")
args = parser.parse_args()


def main():
    global args
    ins = Entry()
    ins.set_model_save_dir(args.model_save_dir)
    ins.set_dataset_dir(args.data_dir)
    ins.set_train_epochs(args.num_epochs)
    ins.set_loss_type(args.loss_type)
    # 1. Build a reader, which yield a sample in the format (data, label)
    #    each time, where data is the decoded image data;
    train_reader = reader.arc_train(args.data_dir,
                                    ins.num_classes)
    # 2. Batch the above samples;
    batched_train_reader = paddle.batch(train_reader,
                                        ins.train_batch_size)
    # 3. Set the reader to use during training to the above batch reader.
    ins.train_reader = batched_train_reader

    ins.train()


if __name__ == "__main__":
    main()
