import os
import sys
from plsc import Entry
from plsc.models import BaseModel
import paddle
import paddle.fluid as fluid
from utils import LogUtil
import numpy as np

CLASS_NUM = 1284213

from data_loader import generate_reader

class UserModel(BaseModel):
    def __init__(self, emb_dim=512):
        self.emb_dim = emb_dim

    def build_network(self,
                      input,
                      is_train=True):
        title_ids = input.title_ids
        content_ids = input.content_ids
        label = input.label
        vob_size = 1841178 + 1
        #embedding layer
        #current shape is [-1, seq_length, emb_dim]
        word_title_sequence_input = fluid.layers.embedding(
            input=title_ids, size=[vob_size, 128], is_sparse=False,
            param_attr=fluid.ParamAttr(name='word_embedding'))
        word_cont_sequence_input = fluid.layers.embedding(
            input=content_ids, size=[vob_size, 128], is_sparse=False,
            param_attr=fluid.ParamAttr(name='word_embedding'))

        #current shape is [-1, emb_dim, seq_length]
        word_title_sequence_input = fluid.layers.transpose(word_title_sequence_input, perm=[0, 2, 1], name='title_transpose')
        word_cont_sequence_input = fluid.layers.transpose(word_cont_sequence_input, perm=[0, 2, 1], name='cont_transpose')

        #current shape is [-1, emb_dim, 1, seq_length], which is NCHW format
        _shape = word_title_sequence_input.shape
        word_title_sequence_input = fluid.layers.reshape(x=word_title_sequence_input,
                    shape=[_shape[0], _shape[1], 1, _shape[2]], inplace=True, name='title_reshape')
        _shape = word_cont_sequence_input.shape
        word_cont_sequence_input = fluid.layers.reshape(x=word_cont_sequence_input,
                    shape=[_shape[0], _shape[1], 1, _shape[2]], inplace=True, name='cont_reshape')

        word_title_win_3 = fluid.layers.conv2d(input=word_title_sequence_input, num_filters=128,
                                            filter_size=(1,3), stride=(1,1), padding=(0,1), act='relu',
                                            name='word_title_win_3_conv')

        word_title_x = fluid.layers.pool2d(input=word_title_win_3, pool_size=(1,4),
                                           pool_type='max', pool_stride=(1,4),
                                           name='word_title_win_3_pool')

        word_cont_win_3 = fluid.layers.conv2d(input=word_cont_sequence_input, num_filters=128,
                                            filter_size=(1,3), stride=(1,1), padding=(0,1), act='relu',
                                            name='word_cont_win_3_conv')

        word_cont_x = fluid.layers.pool2d(input=word_cont_win_3, pool_size=(1,20),
                                           pool_type='max', pool_stride=(1,20), 
                                           name='word_cont_win_3_pool')

        print('word_title_x.shape:', word_title_x.shape)
        print('word_cont_x.shape:', word_cont_x.shape)
        x_concat = fluid.layers.concat(input=[word_title_x, word_cont_x], axis=3, name='feature_concat')
        x_flatten = fluid.layers.flatten(x=x_concat, axis=1, name='feature_flatten')
        x_fc = fluid.layers.fc(input=x_flatten, size=self.emb_dim, act="relu", name='final_fc')
        return x_fc


def train(url2fea_path, topic2fea_path, train_path, val_path, model_save_dir):
    ins = Entry()
    ins.set_with_test(False)
    ins.set_train_epochs(20)
    
    #load id features
    
    word_title_num = 50
    word_cont_num = 1024
    batch_size = int(os.getenv("BATCH_SIZE", "64"))

    input_info = [{'name': 'title_ids',
                   'shape': [-1, word_title_num, 1],
                   'dtype': 'int64'},
                  {'name': 'content_ids',
                   'shape': [-1, word_cont_num, 1],
                   'dtype': 'int64'},
                  {'name': 'label',
                   'shape': [-1, 1],
                   'dtype': 'int64'}
                 ]
    ins.set_input_info(input_info)
    ins.set_class_num(CLASS_NUM)

    emb_dim = int(os.getenv("EMB_DIM", "512"))
    model = UserModel(emb_dim=emb_dim)
    ins.set_model(model)
    ins.set_train_batch_size(batch_size)

    sgd_optimizer = fluid.optimizer.Adam(learning_rate=1e-3)
    ins.set_optimizer(sgd_optimizer)

    train_reader = generate_reader(None, None, train_path)
    ins.train_reader = train_reader
    
    ins.set_train_epochs(20)
    ins.set_model_save_dir("./saved_model")
    ins.set_loss_type('dist_softmax')
    ins.train()



if __name__ == "__main__":
    data = './package/'
    url2fea_path = data + 'click_search_all.url_title_cont.seg.lower.id'
    topic2fea_path = data + 'click_search_all.att.seg.id'
    train_path = data +'train.sample.shuffle.label_expand'
    val_path = data +'test.10w.sample.shuffle.label_expand'
    model_save_dir = data + 'saved_models'
    
    train(url2fea_path, topic2fea_path, train_path, val_path, model_save_dir)

