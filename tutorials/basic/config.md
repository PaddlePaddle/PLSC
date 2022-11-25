# Configuration

PLSC uses [yaml](https://en.wikipedia.org/wiki/YAML) files for unified configuration. 
The aim is to make all experimental results clearly expressed and reproducible. In 
the file, there are several sections, including:

* Global
* FP16
* DistributedStrategy
* Model
* Loss
* Metric
* LRScheduler
* Optimizer
* DataLoader
* Export



## Global

```yaml
# example
Global:
  task_type: recognition
  train_epoch_func: defualt_train_one_epoch
  eval_func: face_verification_eval
  checkpoint: null
  finetune: False
  pretrained_model: null
  output_dir: ./output/
  device: gpu
  save_interval: 1
  max_num_latest_checkpoint: 0
  eval_during_train: True
  eval_interval: 2000
  eval_unit: "step"
  accum_steps: 1
  epochs: 25
  print_batch_step: 100
  use_visualdl: True
  seed: 2022
```

* `task_type`: Task type, currently supports `classification` and `recognition`. Default is `classification`.
* `train_epoch_func`: The training function, usually defined in `plsc/engine/task_type/train.py`. Each task will define a default `defualt_train_one_epoch` function. If the provided training function cannot be satisfied, the user can add a custom training function.
* `eval_func`: Similar to `train_epoch_func`, it is an evaluation function, usually defined in `plsc/engine/task_type/evaluation.py`. Default is `default_eval`.
* `checkpoint`: When training is terminated midway, set the saved checkpoint prefix to resume training, e.g. `output/IResNet50/latest`. Default is `null`.
* `pretrained_model`: Pre-trained weight path prefix, which needs to be set together with the `finetune` parameter. E.g. `output/IResNet50/best_model`. Default is `null`.
* `finetune`: Indicates whether the loaded pretrained weights are for fine-tuning. Default is `False`.
* `output_dir`: Output directory path.
* `device`: Device type, currently only `cpu` and `gpu` are supported.
* `save_interval`: How many `epoch` to save the checkpoint.
* `max_num_latest_checkpoint`: How many recent checkpoints are kept, others will be deleted.
* `eval_during_train`: Indicates whether to evaluate during training.
* `eval_interval`: The frequency of evaluation, which needs to be set together with `eval_unit`.
* `eval_unit`: The unit of evaluation, optional `step` and `epoch`.
* `accum_steps`: Gradient accumulation (merging), when a device stores a batch_size that does not support setting, you can set `accum_steps` > 1 to enable this function. When enabled, divide batch_size into accum_steps runs. This function only works in training mode. The default value is `1`.
* `epochs`: The total epoch of training.
* `print_batch_step`: How many steps to print log once.
* `use_visualdl`: Whether to enable visualdl.
* `seed`: Random number seed.
* `max_train_step`: Maximum training step. When the current number of training steps is greater than the set maximum number of training steps, the training will be stopped early. The default is not set, then ignore this function.
* `flags`: The type is a dictionary representing the FLAGS that need to be set. For example `FLAGS_cudnn_exhaustive_search=0`. The default is not set, then only enable `FLAGS_cudnn_exhaustive_search=1`, `FLAGS_cudnn_batchnorm_spatial_persistent=1`, `FLAGS_max_inplace_grad_add=8`.

## FP16

```yaml
# example
FP16:
  level: O1 # 'O0', 'O1', 'O2'
  fp16_custom_white_list: []
  fp16_custom_black_list: []
  GradScaler:
    init_loss_scaling: 27648.0
    max_loss_scaling: 2.**32
    incr_ratio: 2.0
    decr_ratio: 0.5
    incr_every_n_steps: 1000
    decr_every_n_nan_or_inf: 2
    use_dynamic_loss_scaling: True
    no_unscale_list: ['dist']
```

The FP16 `O0` level is used by default when the FP16 section is not set. The above parameters do not necessarily need to be set explicitly. If they are missing, the default parameter values in the class initialization function will be used.

* `level`: AMP optimization level, optional `O0`, `O1`, `O2`. `O0` means to turn off the AMP function, `O1` means that parameters and gradients use FP32 type, activation uses FP16, `O2` means that parameters, gradients, and activations use FP16. Note that when using O2, the master weight of the parameter is not set here, but is set in the Optimizer section.
* `no_unscale_list`: Provides a special function. If the name set in `no_unscale_list` is in a parameter name, the gradient of this parameter will not be unscaled.

## DistributedStrategy

```yaml
# example
DistributedStrategy:
  data_parallel: True
  data_sharding: False
  recompute:
    layerlist_interval: 1
    names: []
```

Note: Distributed strategy configuration, currently only supports data parallel and recompute.

* `data_parallel`: Whether to use data parallelism.
* `data_sharding`: Whether to use data sharding parallelism. This is mutually exclusive with  `data_parallell`.
* `layerlist_interval`: If `recompute` is set, when there is a `nn.LayerList` layer in the model, you can set `layerlist_interval` to indicate how many blocks to enable recompute
* `names`: If `recompute` is set, when the name in `names` is in a layer's name, this layer will enable recompute. This is mutually exclusive with  `data_parallell`.


## Model

```yaml
# example
Model:
  name: IResNet50
  num_features : 512
  data_format : "NHWC"
  class_num: 93431
  pfc_config:
    sample_ratio: 0.1
    model_parallel: True
```
The `Model` section contains all configuration related to the network model. The configuration of each model may be different, it is recommended to directly see the definition in the model file. The `name` field must be set, and the function or class is instantiated with this string. Other fields are parameters to this function or class initialization function.

## Loss
```yaml
# example
Loss:
  Train:
    - ViTCELoss:
        weight: 1.0
        epsilon: 0.0001
  Eval:
    - CELoss:
        weight: 1.0
```

The `Loss` section contains `Train` and `Eval[optional]` fields. Each field can contain multiple loss functions. For parameters, refer to the definition of the initialization function of the Loss class. Each loss function has a `weight` field, which represents the weight of multiple loss functions.

## Metric
```yaml
# example
Metric:
  Train:
    - TopkAcc:
        topk: [1, 5]
  Eval:
    - TopkAcc:
        topk: [1, 5]
```

The `Metric` section contains `Train` and `Eval[optional]` fields. Each field can contain multiple metric functions. For parameters, refer to the definition of the initialization function of the Metric class.

## LRScheduler

```yaml
# example
LRScheduler:
  name: Step
  boundaries: [10, 16, 22]
  values: [0.2, 0.02, 0.002, 0.0002]
  decay_unit: epoch
```
The `LRScheduler` section contains all configuration related to the learning rate scheduler. The configuration of each `LRScheduler` may be different, it is recommended to directly see the definition in `plsc/scheduler/`. The `name` field must be set, and the function or class is instantiated with this string. Other fields are parameters to this function or class initialization function.


## Optimizer

```yaml
# example
Optimizer:
  name: AdamW
  betas: (0.9, 0.999)
  epsilon: 1e-8
  weight_decay: 0.3
  use_master_param: False
  grad_clip:
    name: ClipGradByGlobalNorm
    clip_norm: 1.0
```
The `Optimizer` section contains all configuration related to the optimizer. The configuration of each `Optimizer` may be different, it is recommended to directly see the definition in `plsc/optimizer/`. The `name` field must be set, and the function or class is instantiated with this string. Other fields are parameters to this function or class initialization function. When instantiating the optimizer, the model parameters are organized in parameter groups.

* `use_master_param`: Indicates whether to use master weight during FP16 `O2` training.
* `grad_clip`: Configuration for gradient clipping. **Note:** Gradient clipping is performed separately for each param group.

## DataLoader

```yaml
# example
DataLoader:
  Train:
    dataset:
      name: FaceIdentificationDataset
      image_root: ./dataset/MS1M_v3/
      cls_label_path: ./dataset/MS1M_v3/label.txt
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.5, 0.5, 0.5]
            std: [0.5, 0.5, 0.5]
            order: ''
        - ToCHWImage: 
    sampler:
      name: DistributedBatchSampler
      batch_size: 128
      drop_last: False
      shuffle: True
    loader:
      num_workers: 8
      use_shared_memory: True

  Eval:
    dataset: 
      name: FaceVerificationDataset
      image_root: ./dataset/MS1M_v3/agedb_30
      cls_label_path: ./dataset/MS1M_v3/agedb_30/label.txt
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.5, 0.5, 0.5]
            std: [0.5, 0.5, 0.5]
            order: ''
        - ToCHWImage:
    sampler:
      name: BatchSampler
      batch_size: 128
      drop_last: False
      shuffle: False
    loader:
      num_workers: 0
      use_shared_memory: True
```

The `DataLoader` section contains `Train` and `Eval` fields.

* `dataset`: The configuration of each `dataset` may be different, it is recommended to directly see the definition in `plsc/data/dataset`. For data preprocessing operations, see `plsc/data/preprocess`.
* `sampler`: In general, `DistributedBatchSampler` can meet the requirements of most data parallelism. If there is an unsatisfied batch sampler, you can add a custom one in `plsc/data/sampler`, e.g. `RepeatedAugSampler`.
* `loader`: Set multi-process configuration for data preprocessing.

## Export

```yaml
# example
Export:
  export_type: onnx
  input_shape: [None, 3, 112, 112]
```

The `Export` section contains the parameter configuration required to export the model.

* `export_type`: The type of the exported model, currently only `paddle` and `onnx` types are supported
* `input_shape`: Specifies the input shape of the exported model.
