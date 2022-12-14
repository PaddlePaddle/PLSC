# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import os
import cv2
import argparse
import numpy as np


def default_preprocess_fn(img,
                          scale=1.0 / 255.0,
                          mean=0.5,
                          std=0.5,
                          swap_rb=True):
    img = (img.astype('float32') * scale - mean) / std
    if swap_rb:
        img = img[:, :, ::-1]
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    return img


class Inference(object):
    def __init__(self,
                 model_type='paddle',
                 model_file=None,
                 params_file=None,
                 preprocess_fn=default_preprocess_fn,
                 postprocess_fn=None):

        assert model_type in ['paddle', 'onnx']
        if model_type == 'paddle':
            assert model_file is not None and os.path.splitext(model_file)[
                1] == '.pdmodel'
            assert params_file is not None and os.path.splitext(params_file)[
                1] == '.pdiparams'

            import paddle.inference as paddle_infer
            config = paddle_infer.Config(model_file, params_file)
            self.predictor = paddle_infer.create_predictor(config)

            input_names = self.predictor.get_input_names()
            self.input_handle = self.predictor.get_input_handle(input_names[0])

            output_names = self.predictor.get_output_names()
            self.output_handle = self.predictor.get_output_handle(output_names[
                0])

        elif model_type == 'onnx':
            assert model_file is not None and os.path.splitext(model_file)[
                1] == '.onnx'
            import onnxruntime
            self.predictor = onnxruntime.InferenceSession(
                model_file,
                providers=[
                    'TensorrtExecutionProvider', 'CUDAExecutionProvider',
                    'CPUExecutionProvider'
                ])
            self.input_name = self.predictor.get_inputs()[0].name

        self.model_type = model_type
        assert preprocess_fn is None or callable(
            preprocess_fn), 'preprocess_fn must be callable'
        assert postprocess_fn is None or callable(
            postprocess_fn), 'preprocess_fn must be callable'
        self.preprocess_fn = preprocess_fn
        self.postprocess_fn = postprocess_fn

    def predict(self, img):

        if self.preprocess_fn is not None:
            img = self.preprocess_fn(img)

        if self.model_type == 'paddle':
            self.input_handle.copy_from_cpu(img)
            self.predictor.run()
            output_data = self.output_handle.copy_to_cpu()

        elif self.model_type == 'onnx':
            output_data = self.predictor.run(None, {self.input_name: img})

        if self.postprocess_fn is not None:
            output_data = self.postprocess_fn(output_data)

        return output_data
