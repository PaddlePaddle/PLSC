from __future__ import print_function
import numpy as np
import argparse

import paddle.fluid as fluid
from plsc import Entry

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_dir",
                    type=str,
                    default=None,
                    help="Directory for checkpoints.")
args = parser.parse_args()

def main():
    global args
    ins = Entry()
    ins.set_checkpoint_dir(args.checkpoint_dir)
    ins.set_loss_type('arcface')
    ins.set_dataset_dir('./data')
    ins.test()

if __name__ == "__main__":
    main()
