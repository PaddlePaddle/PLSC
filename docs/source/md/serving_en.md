# Serving Deployment
## How to export inference model
Generally, the saved model of PLSC only includes parameters, but not the network structure used for prediction. To do prediction, we have to convert the saved model to inference model including both the model structure and parameters.
The following example code shows how to do that.
```python
# export_for_inference.py

from plsc import Entry

if __name__ == "__main__":
    ins = Entry()
    ins.set_checkpoint_dir('./pretrain_model')
    ins.set_model_save_dir('./inference_model')

    ins.convert_for_prediction()
```

The following command can be used to start converting.
```shell script
python export_for_inference.py
```

## How to use inference library
The inference library requires python3.

### How to use the server
Use the following command to install the server.
```shell script
pip3 install plsc-serving
```

Currently, you can only use the server with GPU devices, and it requires the cuda version to be at least 9.0.

You can use the following command to start the server:
```python
from plsc_serving.run import PLSCServer
fs = PLSCServer()
# Set the absolute path for inference model
fs.with_model(model_path = '/XXX/XXX')
fs.run(gpu_index = 0, port = 8010)
```

### How to use the client
First, you have to install ujson:
```shell script
pip install ujson
```

Then, copy the [client script](../../../serving/client/face_service/face_service.py) to your local directory.

You can use the following command to start the client:
```python
from face_service import FaceService
with open('./data/00000000.jpg', 'rb') as f:
    image = f.read()
fc = FaceService()
# connect to the server
fc.connect('127.0.0.1:8010')
result = fc.encode([image])
print(result[0])
fc.close()
```