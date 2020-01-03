from __future__ import print_function
import numpy as np

import paddle.fluid as fluid
from plsc import Entry


def main():
    ins = Entry()
    ins.set_checkpoint_dir('./saved_model/5')
    ins.set_train_batch_size(1)
    ins.set_dataset_dir("./data")
    ins.predict()

if __name__ == "__main__":
    main()
