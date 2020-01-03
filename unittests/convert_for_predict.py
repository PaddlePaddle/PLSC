from __future__ import print_function
import numpy as np

import paddle.fluid as fluid
from plsc import Entry


def main():
    ins = Entry()
    ins.set_checkpoint_dir('./saved_model/5')
    ins.set_model_save_dir('./output_infer')
    ins.convert_for_prediction()

if __name__ == "__main__":
    main()
