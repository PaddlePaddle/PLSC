## Install PaddlePaddle from whl Package
```
# [optional] modify cuda version, e.g. post112 to post116
# require python==3.7, paddlepaddle-gpu==2.4.x
python -m pip install paddlepaddle-gpu==2.4.0.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

## Install PaddlePaddle from Source Code

For more install information, ref to [PaddlePaddle](https://www.paddlepaddle.org.cn/)

```shell

git clone https://github.com/PaddlePaddle/Paddle.git

cd /path/to/Paddle/

git checkout -b release/2.4 remotes/origin/release/2.4

mkdir build && cd build

cmake .. -DWITH_TESTING=OFF -DWITH_GPU=ON -DWITH_GOLANG=OFF -DWITH_STYLE_CHECK=ON -DCMAKE_INSTALL_PREFIX=$PWD/output -DWITH_DISTRIBUTE=ON -DCMAKE_BUILD_TYPE=Release -DPY_VERSION=3.7

make -j20 && make install -j20

pip install output/opt/paddle/share/wheels/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl

```

## Install PLSC

```shell
git clone https://github.com/PaddlePaddle/PLSC.git

cd /path/to/PLSC/
# [optional] pip install -r requirements.txt
python setup.py develop
```
