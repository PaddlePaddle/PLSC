# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from setuptools import setup, find_packages

setup(name="plsc",
      version="0.1.0",
      description="Large Scale Classfication via distributed fc.",
      author='lilong',
      author_email="lilong.albert@gmail.com",
      url="http",
      license="Apache",
      #packages=['paddleXML'],
      packages=find_packages(),
      install_requires=['paddlepaddle>=1.6.2', 'sklearn', 'easydict'],
      python_requires='>=2'
     )
