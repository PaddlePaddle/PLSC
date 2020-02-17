# This demo shows how to use user-defined training dataset.
# The following steps are needed to use user-defined training datasets:
# 1. Build a reader, which preprocess images and yield a sample in the
#    format (data, label) each time, where data is the decoded image data;
# 2. Batch the above samples;
# 3. Set the reader to use during training to the above batch reader.

import argparse

import paddle
from plsc import Entry
from plsc.utils import dali

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
    # 1. Build a dali reader
    gpu_id = ins.trainer_id % 8 # Assume 8 card per machine
    dali_iter = dali.train(ins.train_batch_size,
                           args.data_dir,
                           file_list="label.txt",
                           trainer_id=ins.trainer_id,
                           trainers_num=ins.num_trainers,
                           gpu_id=gpu_id,
                           data_layout=ins.data_format
                           )
    # 2. Set use_dali to True;
    ins.set_use_dali()
    # 3. Set the reader to use during training to the above reader.
    ins.train_reader = dali_iter

    ins.train()


if __name__ == "__main__":
    main()
