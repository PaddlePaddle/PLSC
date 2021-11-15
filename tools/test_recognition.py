# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os
import argparse
import requests
import logging
import imghdr
import pickle
import tarfile
from functools import partial

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from prettytable import PrettyTable
from PIL import Image, ImageDraw, ImageFont
import paddle
from paddle.inference import Config
from paddle.inference import create_predictor

__all__ = ["InsightFace", "parser"]

# Ref: https://github.com/littletomatodonkey/insight-face-paddle/blob/main/insightface_paddle.py


def parser(add_help=True):
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument(
        "--det", action="store_true", help="Whether to detect.")
    parser.add_argument(
        "--rec", action="store_true", help="Whether to recognize.")

    parser.add_argument(
        "--det_model_file_path",
        type=str,
        default="models/blazeface_fpn_ssh_1000e_v1.0_infer/inference.pdmodel",
        help="The detection model file path.")
    parser.add_argument(
        "--det_params_file_path",
        type=str,
        default="models/blazeface_fpn_ssh_1000e_v1.0_infer/inference.pdiparams",
        help="The detection params file path.")
    parser.add_argument(
        "--rec_model_file_path",
        type=str,
        default="models/ms1mv3_r50_static_128_fp16_0.1_epoch_24_infer/FresResNet50.pdmodel",
        help="The detection model file path.")
    parser.add_argument(
        "--rec_params_file_path",
        type=str,
        default="models/ms1mv3_r50_static_128_fp16_0.1_epoch_24_infer/FresResNet50.pdiparams",
        help="The detection params file path.")
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether use GPU to predict. Default by True.")
    parser.add_argument(
        "--enable_mkldnn",
        type=str2bool,
        default=True,
        help="Whether use MKLDNN to predict, valid only when --use_gpu is False. Default by False."
    )
    parser.add_argument(
        "--cpu_threads",
        type=int,
        default=1,
        help="The num of threads with CPU, valid only when --use_gpu is False. Default by 1."
    )
    parser.add_argument(
        "--input",
        type=str,
        help="The path or directory of image(s) or video to be predicted.")
    parser.add_argument(
        "--output",
        type=str,
        default="./output/",
        help="The directory of prediction result.")
    parser.add_argument(
        "--det_thresh",
        type=float,
        default=0.8,
        help="The threshold of detection postprocess. Default by 0.8.")
    parser.add_argument(
        "--index", type=str, default=None, help="The path of index file.")
    parser.add_argument(
        "--cdd_num",
        type=int,
        default=10,
        help="The number of candidates in the recognition retrieval. Default by 10."
    )
    parser.add_argument(
        "--rec_thresh",
        type=float,
        default=0.45,
        help="The threshold of recognition postprocess. Default by 0.45.")
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="The maxium of batch_size to recognize. Default by 1.")
    parser.add_argument(
        "--build_index",
        type=str,
        default=None,
        help="The path of index to be build.")
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help="The img(s) dir used to build index.")
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="The label file path used to build index.")

    return parser


def print_config(args):
    args = vars(args)
    table = PrettyTable(['Param', 'Value'])
    for param in args:
        table.add_row([param, args[param]])
    width = len(str(table).split("\n")[0])
    print("{}".format("-" * width))
    print("PaddleFace".center(width))
    print(table)
    print("Powered by PaddlePaddle!".rjust(width))
    print("{}".format("-" * width))


def normalize_image(img, scale=None, mean=None, std=None, order='chw'):
    if isinstance(scale, str):
        scale = eval(scale)
    scale = np.float32(scale if scale is not None else 1.0 / 255.0)
    mean = mean if mean is not None else [0.485, 0.456, 0.406]
    std = std if std is not None else [0.229, 0.224, 0.225]

    shape = (3, 1, 1) if order == 'chw' else (1, 1, 3)
    mean = np.array(mean).reshape(shape).astype('float32')
    std = np.array(std).reshape(shape).astype('float32')

    if isinstance(img, Image.Image):
        img = np.array(img)

    assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
    return (img.astype('float32') * scale - mean) / std


def to_CHW_image(img):
    if isinstance(img, Image.Image):
        img = np.array(img)
    return img.transpose((2, 0, 1))


class ColorMap(object):
    def __init__(self, num):
        super().__init__()
        self.get_color_map_list(num)
        self.color_map = {}
        self.ptr = 0

    def __getitem__(self, key):
        return self.color_map[key]

    def update(self, keys):
        for key in keys:
            if key not in self.color_map:
                i = self.ptr % len(self.color_list)
                self.color_map[key] = self.color_list[i]
                self.ptr += 1

    def get_color_map_list(self, num_classes):
        color_map = num_classes * [0, 0, 0]
        for i in range(0, num_classes):
            j = 0
            lab = i
            while lab:
                color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
                color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
                color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
                j += 1
                lab >>= 3
        self.color_list = [
            color_map[i:i + 3] for i in range(0, len(color_map), 3)
        ]


class ImageReader(object):
    def __init__(self, inputs):
        super().__init__()
        self.idx = 0
        if isinstance(inputs, np.ndarray):
            self.image_list = [inputs]
        else:
            imgtype_list = {'jpg', 'bmp', 'png', 'jpeg', 'rgb', 'tif', 'tiff'}
            self.image_list = []
            if os.path.isfile(inputs):
                if imghdr.what(inputs) not in imgtype_list:
                    raise Exception(
                        f"Error type of input path, only support: {imgtype_list}"
                    )
                self.image_list.append(inputs)
            elif os.path.isdir(inputs):
                tmp_file_list = os.listdir(inputs)
                warn_tag = False
                for file_name in tmp_file_list:
                    file_path = os.path.join(inputs, file_name)
                    if not os.path.isfile(file_path):
                        warn_tag = True
                        continue
                    if imghdr.what(file_path) in imgtype_list:
                        self.image_list.append(file_path)
                    else:
                        warn_tag = True
                if warn_tag:
                    logging.warning(
                        f"The directory of input contine directory or not supported file type, only support: {imgtype_list}"
                    )
            else:
                raise Exception(
                    f"The file of input path not exist! Please check input: {inputs}"
                )

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= len(self.image_list):
            raise StopIteration

        data = self.image_list[self.idx]
        if isinstance(data, np.ndarray):
            self.idx += 1
            return data, "tmp.png"
        path = data
        _, file_name = os.path.split(path)
        img = cv2.imread(path)
        if img is None:
            logging.warning(f"Error in reading image: {path}! Ignored.")
            self.idx += 1
            return self.__next__()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.idx += 1
        return img, file_name

    def __len__(self):
        return len(self.image_list)


class VideoReader(object):
    def __init__(self, inputs):
        super().__init__()
        videotype_list = {"mp4"}
        if os.path.splitext(inputs)[-1][1:] not in videotype_list:
            raise Exception(
                f"The input file is not supported, only support: {videotype_list}"
            )
        if not os.path.isfile(inputs):
            raise Exception(
                f"The file of input path not exist! Please check input: {inputs}"
            )
        self.capture = cv2.VideoCapture(inputs)
        self.file_name = os.path.split(inputs)[-1]

    def get_info(self):
        info = {}
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
        info["file_name"] = self.file_name
        info["fps"] = 30
        info["shape"] = (width, height)
        info["fourcc"] = cv2.VideoWriter_fourcc(* 'mp4v')
        return info

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.capture.read()
        if not ret:
            raise StopIteration
        return frame, self.file_name


class ImageWriter(object):
    def __init__(self, output_dir):
        super().__init__()
        if output_dir is None:
            raise Exception(
                "Please specify the directory of saving prediction results by --output."
            )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def write(self, image, file_name):
        path = os.path.join(self.output_dir, file_name)
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


class VideoWriter(object):
    def __init__(self, output_dir, video_info):
        super().__init__()
        if output_dir is None:
            raise Exception(
                "Please specify the directory of saving prediction results by --output."
            )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, video_info["file_name"])
        fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
        self.writer = cv2.VideoWriter(output_path, video_info["fourcc"],
                                      video_info["fps"], video_info["shape"])

    def write(self, frame, file_name):
        self.writer.write(frame)

    def __del__(self):
        if hasattr(self, "writer"):
            self.writer.release()


class BasePredictor(object):
    def __init__(self, predictor_config):
        super().__init__()
        self.predictor_config = predictor_config
        self.predictor, self.input_names, self.output_names = self.load_predictor(
            predictor_config["model_file"], predictor_config["params_file"])

    def load_predictor(self, model_file, params_file):
        config = Config(model_file, params_file)
        if self.predictor_config["use_gpu"]:
            config.enable_use_gpu(200, 0)
            config.switch_ir_optim(True)
        else:
            config.disable_gpu()
            config.set_cpu_math_library_num_threads(self.predictor_config[
                "cpu_threads"])

            if self.predictor_config["enable_mkldnn"]:
                try:
                    # cache 10 different shapes for mkldnn to avoid memory leak
                    config.set_mkldnn_cache_capacity(10)
                    config.enable_mkldnn()
                except Exception as e:
                    logging.error(
                        "The current environment does not support `mkldnn`, so disable mkldnn."
                    )
        config.disable_glog_info()
        config.enable_memory_optim()
        # use zero copy
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)
        input_names = predictor.get_input_names()
        output_names = predictor.get_output_names()
        return predictor, input_names, output_names

    def preprocess(self):
        raise NotImplementedError

    def postprocess(self):
        raise NotImplementedError

    def predict(self, img):
        raise NotImplementedError


class Detector(BasePredictor):
    def __init__(self, det_config, predictor_config):
        super().__init__(predictor_config)
        self.det_config = det_config
        self.target_size = self.det_config["target_size"]
        self.thresh = self.det_config["thresh"]

    def preprocess(self, img):
        resize_h, resize_w = self.target_size
        img_shape = img.shape
        img_scale_x = resize_w / img_shape[1]
        img_scale_y = resize_h / img_shape[0]
        img = cv2.resize(
            img, None, None, fx=img_scale_x, fy=img_scale_y, interpolation=1)
        img = normalize_image(
            img,
            scale=1. / 255.,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            order='hwc')
        img_info = {}
        img_info["im_shape"] = np.array(
            img.shape[:2], dtype=np.float32)[np.newaxis, :]
        img_info["scale_factor"] = np.array(
            [img_scale_y, img_scale_x], dtype=np.float32)[np.newaxis, :]

        img = img.transpose((2, 0, 1)).copy()
        img_info["image"] = img[np.newaxis, :, :, :]
        return img_info

    def postprocess(self, np_boxes):
        expect_boxes = (np_boxes[:, 1] > self.thresh) & (np_boxes[:, 0] > -1)
        return np_boxes[expect_boxes, :]

    def predict(self, img):
        inputs = self.preprocess(img)
        for input_name in self.input_names:
            input_tensor = self.predictor.get_input_handle(input_name)
            input_tensor.copy_from_cpu(inputs[input_name])
        self.predictor.run()
        output_tensor = self.predictor.get_output_handle(self.output_names[0])
        np_boxes = output_tensor.copy_to_cpu()
        # boxes_num = self.detector.get_output_handle(self.detector_output_names[1])
        # np_boxes_num = boxes_num.copy_to_cpu()
        box_list = self.postprocess(np_boxes)
        return box_list


class Recognizer(BasePredictor):
    def __init__(self, rec_config, predictor_config):
        super().__init__(predictor_config)
        if rec_config["index"] is not None:
            self.load_index(rec_config["index"])
        self.rec_config = rec_config
        self.cdd_num = self.rec_config["cdd_num"]
        self.thresh = self.rec_config["thresh"]
        self.max_batch_size = self.rec_config["max_batch_size"]

    def preprocess(self, img, box_list=None):
        img = normalize_image(
            img,
            scale=1. / 255.,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            order='hwc')
        if box_list is None:
            height, width = img.shape[:2]
            box_list = [np.array([0, 0, 0, 0, width, height])]
        batch = []
        input_batches = []
        cnt = 0
        for idx, box in enumerate(box_list):
            box[box < 0] = 0
            xmin, ymin, xmax, ymax = list(map(int, box[2:]))
            face_img = img[ymin:ymax, xmin:xmax, :]
            face_img = cv2.resize(face_img, (112, 112)).transpose(
                (2, 0, 1)).copy()
            batch.append(face_img)
            cnt += 1
            if cnt % self.max_batch_size == 0 or (idx + 1) == len(box_list):
                input_batches.append(np.array(batch))
                batch = []
        return input_batches

    def postprocess(self):
        pass

    def retrieval(self, np_feature):
        labels = []
        for feature in np_feature:
            similarity = cosine_similarity(self.index_feature,
                                           feature).squeeze()
            abs_similarity = np.abs(similarity)
            candidate_idx = np.argpartition(abs_similarity,
                                            -self.cdd_num)[-self.cdd_num:]
            remove_idx = np.where(abs_similarity[candidate_idx] < self.thresh)
            candidate_idx = np.delete(candidate_idx, remove_idx)
            candidate_label_list = list(np.array(self.label)[candidate_idx])
            if len(candidate_label_list) == 0:
                maxlabel = ""
            else:
                maxlabel = max(candidate_label_list,
                               key=candidate_label_list.count)
            labels.append(maxlabel)
        return labels

    def load_index(self, file_path):
        with open(file_path, "rb") as f:
            index = pickle.load(f)
        self.label = index["label"]
        self.index_feature = np.array(index["feature"]).squeeze()

    def predict(self, img, box_list=None):
        batch_list = self.preprocess(img, box_list)
        feature_list = []
        for batch in batch_list:
            for input_name in self.input_names:
                input_tensor = self.predictor.get_input_handle(input_name)
                input_tensor.copy_from_cpu(batch)
            self.predictor.run()
            output_tensor = self.predictor.get_output_handle(self.output_names[
                0])
            np_feature = output_tensor.copy_to_cpu()
            feature_list.append(np_feature)
        return np.array(feature_list)


class InsightFace(object):
    def __init__(self, args, print_info=True):
        super().__init__()
        if print_info:
            print_config(args)

        self.font_path = "assets/SourceHanSansCN-Medium.otf"
        self.args = args

        predictor_config = {
            "use_gpu": args.use_gpu,
            "enable_mkldnn": args.enable_mkldnn,
            "cpu_threads": args.cpu_threads
        }
        if args.det:
            det_config = {"thresh": args.det_thresh, "target_size": [640, 640]}
            predictor_config["model_file"] = args.det_model_file_path
            predictor_config["params_file"] = args.det_params_file_path
            self.det_predictor = Detector(det_config, predictor_config)
            self.color_map = ColorMap(100)

        if args.rec:
            rec_config = {
                "max_batch_size": args.max_batch_size,
                "resize": 112,
                "thresh": args.rec_thresh,
                "index": args.index,
                "cdd_num": args.cdd_num
            }
            predictor_config["model_file"] = args.rec_model_file_path
            predictor_config["params_file"] = args.rec_params_file_path
            self.rec_predictor = Recognizer(rec_config, predictor_config)

    def preprocess(self, img):
        img = img.astype(np.float32, copy=False)
        return img

    def draw(self, img, box_list, labels):
        self.color_map.update(labels)
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)

        for i, dt in enumerate(box_list):
            bbox, score = dt[2:], dt[1]
            label = labels[i]
            color = tuple(self.color_map[label])

            xmin, ymin, xmax, ymax = bbox

            font_size = max(int((xmax - xmin) // 6), 10)
            font = ImageFont.truetype(self.font_path, font_size)

            text = "{} {:.4f}".format(label, score)
            th = sum(font.getmetrics())
            tw = font.getsize(text)[0]
            start_y = max(0, ymin - th)

            draw.rectangle(
                [(xmin, start_y), (xmin + tw + 1, start_y + th)], fill=color)
            draw.text(
                (xmin + 1, start_y),
                text,
                fill=(255, 255, 255),
                font=font,
                anchor="la")
            draw.rectangle(
                [(xmin, ymin), (xmax, ymax)], width=2, outline=color)
        return np.array(im)

    def predict_np_img(self, img):
        input_img = self.preprocess(img)
        box_list = None
        np_feature = None
        if hasattr(self, "det_predictor"):
            box_list = self.det_predictor.predict(input_img)
        if hasattr(self, "rec_predictor"):
            np_feature = self.rec_predictor.predict(input_img, box_list)
        return box_list, np_feature

    def init_reader_writer(self, input_data):
        if isinstance(input_data, np.ndarray):
            self.input_reader = ImageReader(input_data)
            if hasattr(self, "det_predictor"):
                self.output_writer = ImageWriter(self.args.output)
        elif isinstance(input_data, str):
            if input_data.endswith('mp4'):
                self.input_reader = VideoReader(input_data)
                info = self.input_reader.get_info()
                self.output_writer = VideoWriter(self.args.output, info)
            else:
                self.input_reader = ImageReader(input_data)
                if hasattr(self, "det_predictor"):
                    self.output_writer = ImageWriter(self.args.output)
        else:
            raise Exception(
                f"The input data error. Only support path of image or video(.mp4) and dirctory that include images."
            )

    def predict(self, input_data, print_info=False):
        """Predict input_data.

        Args:
            input_data (str | NumPy.array): The path of image, or the derectory including images, or the image data in NumPy.array format.
            print_info (bool, optional): Wheather to print the prediction results. Defaults to False.

        Yields:
            dict: {
                "box_list": The prediction results of detection.
                "features": The output of recognition.
                "labels": The results of retrieval.
                }
        """
        self.init_reader_writer(input_data)
        for img, file_name in self.input_reader:
            if img is None:
                logging.warning(f"Error in reading img {file_name}! Ignored.")
                continue
            box_list, np_feature = self.predict_np_img(img)
            if np_feature is not None:
                labels = self.rec_predictor.retrieval(np_feature)
            else:
                labels = ["face"] * len(box_list)
            if box_list is not None:
                result = self.draw(img, box_list, labels=labels)
                self.output_writer.write(result, file_name)
            if print_info:
                logging.info(f"File: {file_name}, predict label(s): {labels}")
            yield {
                "box_list": box_list,
                "features": np_feature,
                "labels": labels
            }
        logging.info(f"Predict complete!")

    def build_index(self):
        img_dir = self.args.img_dir
        label_path = self.args.label
        with open(label_path, "r") as f:
            sample_list = f.readlines()

        feature_list = []
        label_list = []

        for idx, sample in enumerate(sample_list):
            name, label = sample.strip().split("\t")
            img = cv2.imread(os.path.join(img_dir, name))
            if img is None:
                logging.warning(f"Error in reading img {name}! Ignored.")
                continue
            box_list, np_feature = self.predict_np_img(img)
            feature_list.append(np_feature[0])
            label_list.append(label)

            if idx % 100 == 0:
                logging.info(f"Build idx: {idx}")

        with open(self.args.build_index, "wb") as f:
            pickle.dump({"label": label_list, "feature": feature_list}, f)
        logging.info(
            f"Build done. Total {len(label_list)}. Index file has been saved in \"{self.args.build_index}\""
        )


# for CLI
def main(args=None):
    logging.basicConfig(level=logging.INFO)

    args = parser().parse_args()
    predictor = InsightFace(args)
    if args.build_index:
        predictor.build_index()
    else:
        res = predictor.predict(args.input, print_info=True)
        for _ in res:
            pass


if __name__ == "__main__":
    main()
