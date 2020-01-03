from __future__ import print_function
import numpy as np
import sys
import argparse

import paddle.fluid as fluid
from plsc import Entry
from plsc.models.resnet import ResNet50


parser = argparse.ArgumentParser()
parser.add_argument("--model_save_dir",
                    type=str,
                    default="./saved_model",
                    help="Directory to save models.")
parser.add_argument("--checkpoint_dir",
                    type=str,
                    default=None,
                    help="Directory for checkpoints.")
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
parser.add_argument("--fs_name",
                    type=str,
                    default=None,
                    help="fs_name for hdfs.")
parser.add_argument("--fs_ugi",
                    type=str,
                    default=None,
                    help="fs_ugi for hdfs.")
parser.add_argument("--fs_dir_load",
                    type=str,
                    default=None,
                    help="Remote directory for hdfs to load from")
parser.add_argument("--fs_dir_save",
                    type=str,
                    default=None,
                    help="Remote directory for hdfs to save to")
args = parser.parse_args()

def main():
    global args
    ins = Entry()
    ins.set_model_save_dir(args.model_save_dir)
    ins.set_dataset_dir(args.data_dir)
    ins.set_train_epochs(args.num_epochs)
    ins.set_checkpoint_dir(args.checkpoint_dir)
    ins.set_loss_type(args.loss_type)
    if args.fs_name:
        ins.set_hdfs_info(args.fs_name,
                          args.fs_ugi,
                          args.fs_dir_save,
                          args.fs_dir_load)
    ins.train()

if __name__ == "__main__":
    main()
