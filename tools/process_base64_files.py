#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import division
from __future__ import print_function

import argparse
import logging
import math
import os
import random
import sqlite3
import tempfile
import time

import six

logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger()

parser = argparse.ArgumentParser(description="""
    Tool to preprocess dataset in base64 format.""")

"""
We assume that the directory of dataset contains a file-list file, and one 
or more data files. Each line of the file-list file represents a data file.
Each line of a data file represents a image in base64 format.

For example:

dir
  |-- file_list.txt
  |-- part_one.txt
  `-- part_two.txt

In the above example, the file file_list.txt has two lines:

    part_one.txt
    part_two.txt

Each line in part_one.txt and part_two.txt represents a image in base64
format.
"""

parser.add_argument("--data_dir",
                    type=str,
                    required=True,
                    default=None,
                    help="Directory for datasets.")
parser.add_argument("--file_list",
                    type=str,
                    required=True,
                    default=None,
                    help="The file contains a set of data files.")
parser.add_argument("--nranks",
                    type=int,
                    required=True,
                    default=1,
                    help="Number of ranks.")
args = parser.parse_args()


class Base64Preprocessor(object):
    def __init__(self, data_dir, file_list, nranks):
        super(Base64Preprocessor, self).__init__()
        self.data_dir = data_dir
        self.file_list = file_list
        self.nranks = nranks

        self.tempfile = tempfile.NamedTemporaryFile(delete=False, dir=data_dir)
        self.sqlite3_file = self.tempfile.name
        self.conn = None
        self.cursor = None

    def insert_to_db(self, cnt, line):
        label = int(line[0])
        data = line[1]
        sql_cmd = "INSERT INTO DATASET (ID, DATA, LABEL) "
        sql_cmd += "VALUES ({}, '{}', {});".format(cnt, data, label)
        self.cursor.execute(sql_cmd)

    def create_db(self):
        start = time.time()
        print(self.sqlite3_file)
        self.conn = sqlite3.connect(self.sqlite3_file)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''CREATE TABLE DATASET
                             (ID INT PRIMARY KEY    NOT NULL,
                              DATA          TEXT    NOT NULL,
                              LABEL         INT     NOT NULL);''')

        file_list_path = os.path.join(self.data_dir, self.file_list)
        with open(file_list_path, 'r') as f:
            cnt = 0
            if six.PY2:
                for line in f.xreadlines():
                    line = line.strip()
                    file_path = os.path.join(self.data_dir, line)
                    with open(file_path, 'r') as df:
                        for line_local in df.xreadlines():
                            line_local = line_local.strip().split('\t')
                            self.insert_to_db(cnt, line_local)
                            cnt += 1
                    os.remove(file_path)
            else:
                for line in f:
                    line = line.strip()
                    file_path = os.path.join(self.data_dir, line)
                    with open(file_path, 'r') as df:
                        for line_local in df:
                            line_local = line_local.strip().split('\t')
                            self.insert_to_db(cnt, line_local)
                            cnt += 1
                    os.remove(file_path)

        self.conn.commit()
        diff = time.time() - start
        print("time: ", diff)
        return cnt

    def shuffle_files(self):
        num = self.create_db()
        nranks = self.nranks
        index = [i for i in range(num)]

        seed = int(time.time())
        random.seed(seed)
        random.shuffle(index)

        start_time = time.time()

        lines_per_rank = int(math.ceil(num / nranks))
        total_num = lines_per_rank * nranks
        index = index + index[0:total_num - num]
        assert len(index) == total_num

        for rank in range(nranks):
            start = rank * lines_per_rank
            end = (rank + 1) * lines_per_rank  # exclusive
            f_handler = open(os.path.join(self.data_dir,
                                          ".tmp_" + str(rank)), 'w')
            for i in range(start, end):
                idx = index[i]
                sql_cmd = "SELECT DATA, LABEL FROM DATASET WHERE ID={};".format(
                    idx)
                cursor = self.cursor.execute(sql_cmd)
                for result in cursor:
                    data = result[0]
                    label = result[1]
                    line = data + '\t' + str(label) + '\n'
                f_handler.writelines(line)
            f_handler.close()

        data_dir = self.data_dir
        file_list = self.file_list
        file_list = os.path.join(data_dir, file_list)
        temp_file_list = file_list + "_temp"
        with open(temp_file_list, 'w') as f_t:
            for rank in range(nranks):
                line = "base64_rank_{}".format(rank)
                line += '\n'
                f_t.writelines(line)
                os.rename(os.path.join(data_dir, ".tmp_" + str(rank)),
                          os.path.join(data_dir, "base64_rank_{}".format(rank)))

        os.remove(file_list)
        os.rename(temp_file_list, file_list)
        print("shuffle time: ", time.time() - start_time)

    def close_db(self):
        self.conn.close()
        self.tempfile.close()
        os.remove(self.sqlite3_file)


def main():
    global args

    obj = Base64Preprocessor(args.data_dir, args.file_list, args.nranks)
    obj.shuffle_files()
    obj.close_db()


if __name__ == "__main__":
    main()
