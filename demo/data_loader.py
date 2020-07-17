import numpy as np
import sys
import os

word_title_num = 50
word_cont_num = 1024
word_att_num = 10
CLASS_NUM = 1284213

def pad_and_trunk(_list, fix_sz = -1):
    if len(_list) > 0 and _list[0] == '':
        _list = []
    _list = _list[:fix_sz]
    if len(_list) < fix_sz:
        pad = ['0' for i in range(fix_sz - len(_list))]
        _list.extend(pad)
    return _list

def generate_reader(url2fea, topic2fea, _path, class_num=CLASS_NUM):

    def reader():
        print 'file open.'
        trainer_id = int(os.getenv("PADDLE_TRAINER_ID", "0"))
        if os.getenv("PADDLE_TRAINER_ENDPOINTS"):
            trainer_count = len(os.getenv("PADDLE_TRAINER_ENDPOINTS").split(","))
        else:
            trainer_count = int(os.getenv("PADDLE_TRAINERS", "1"))
        f = open(_path)
        sample_index = 0
        for line in f:
            line = line.strip('\n')
            if len(line) == 0:
                continue
            
            part = line.split('\t')

            url  = part[0]
            title_ids = part[1]
            content_ids = part[2]
            label = int(part[3])

            if sample_index % trainer_count != trainer_id:
                sample_index += 1
                continue
            sample_index += 1

            title_ids = pad_and_trunk(title_ids.split(','), word_title_num)
            content_ids = pad_and_trunk(content_ids.split(','), word_cont_num)

            title_input_x_train = np.asarray(title_ids, dtype='int64').reshape( (len(title_ids), 1) )
            content_input_x_train = np.asarray(content_ids, dtype='int64').reshape( (len(content_ids), 1) )

            label = np.array([label])
            yield title_input_x_train, content_input_x_train, label
    
        f.close()
        print 'file close.'
    return reader

if __name__ == '__main__':
    
    #load_validation(url2fea, topic2fea, './data_makeup/merge_att_data/format_sample_v1/test.sample.shuffle')
    
    '''
    for (x1, x2, x3, y) in generate_batch_from_file(url2fea, topic2fea, \
                    './data_makeup/merge_att_data/format_sample_v1/train.sample.shuffle', 50):
        print x1[0], x2[0], x3[0], y[0]
        break
    '''

    for x1, x2, x3, x4 in generate_reader(None, None, './data_makeup/merge_att_data/format_sample_v4/test.10w.sample.shuffle').reader():
        print x1, x2, x3, x4
        break

