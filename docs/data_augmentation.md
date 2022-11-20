# Data Augmentation


### AutoAugment

| timm AutoAugment | Help                                                         | plsc TimmAutoAugment |
| ---------------- | ------------------------------------------------------------ | -------------------- |
| aa               | Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1) | config_str           |
|                  | Interpolation mode                                           | interpolation        |
|                  | Image size to apply resize                                   | img_size             |
|                  | fillcolor                                                    | mean                 |

### Mixup and Cutmix
| timm Mixup      | help                                                         | plsc Mixup | plsc Cutmix                   |
| --------------- | ------------------------------------------------------------ | ---------- | ----------------------------- |
| mixup_alpha     | mixup alpha value, mixup is active if > 0.                   | alpha      |                               |
| cutmix_alpha    | cutmix alpha value, cutmix is active if > 0.                 |            | alpha                         |
| cutmix_minmax   | cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None. |            | does not support              |
| prob            | probability of applying mixup or cutmix per batch or element |            | Does not need, always 1.0     |
| switch_prob     | probability of switching to cutmix instead of mixup when both are active | prob       | prob                          |
| mode            | how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element) |            | Does not need, always 'batch' |
| correct_lam     | apply lambda correction when cutmix bbox clipped by image borders |            | Does not need, always True    |
| label_smoothing | apply label smoothing to the mixed target tensor             | epsilon    | epsilon                       |
| num_classes     | number of classes for target                                 | class_num  | class_num                     |


### RandomErasing

| timm RandomErasing | help                                                         | plsc RandomErasing      |
| ------------------ | ------------------------------------------------------------ | ----------------------- |
| probability        | Probability that the Random Erasing operation will be performed. | EPSILON                 |
| min_area           | Minimum percentage of erased area wrt input image area.      | sl                      |
| max_area           | Maximum percentage of erased area wrt input image area.      | sh                      |
| min_aspect         | Minimum aspect ratio of erased area.                         | r1                      |
| mode               | pixel color mode, one of 'const', 'rand', or 'pixel'         | mode                    |
| min_count          | minimum number of erasing blocks per image, area per box is scaled by count. | Does not need, always 1 |
| max_count          | maximum number of erasing blocks per image, area per box is scaled by count. per-image count is randomly chosen between 1 and this value. | Does not need, always 1 |
|                    | the number of try to random erase                            | attempt                 |


### Example configuration in yaml

```yaml
DataLoader:
  Train:
    dataset:
      name: ImageNetDataset
      image_root: ./dataset/ILSVRC2012/
      cls_label_path: ./dataset/ILSVRC2012/train_list.txt
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
            interpolation: bicubic
            backend: pil
        - RandFlipImage:
            flip_code: 1
        - TimmAutoAugment:
            config_str: rand-m9-mstd0.5-inc1
            interpolation: bicubic
            img_size: 224
            mean: [0.485, 0.456, 0.406]
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
        - RandomErasing:
            EPSILON: 0.25
            sl: 0.02
            sh: 1.0/3.0
            r1: 0.3
            attempt: 10
            use_log_aspect: True
            mode: pixel
        - ToCHWImage:
      batch_transform_ops:
        - TransformOpSampler:
            Mixup:
              alpha: 0.8
              prob: 0.5
              epsilon: 0.1
              class_num: 1000
            Cutmix:
              alpha: 1.0
              prob: 0.5
              epsilon: 0.1
              class_num: 1000
    sampler:
      name: RepeatedAugSampler
      batch_size: 64
      drop_last: False
      shuffle: True
    loader:
      num_workers: 8
      use_shared_memory: True
```

### Example code in py

```python

transform_train = transforms.Compose([
        transforms.DecodeImage(to_rgb=True, channel_first=False),
        transforms.RandCropImage(args.input_size, interpolation="bicubic"),  # 3 is bicubic
        transforms.RandFlipImage(),
        transforms.TimmAutoAugment(config_str=args.aa, interpolation="bicubic", img_size=args.input_size, mean=[0.485, 0.456, 0.406]),
        transforms.NormalizeImage(scale=1.0/255.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], order='hwc'),
        transforms.RandomErasing(EPSILON=args.reprob, sl=0.02, sh=1.0/3.0, r1=0.3, attempt=10, use_log_aspect=True, mode=args.remode),
        transforms.ToCHWImage()])


batch_transform_ops = {}
batch_transform_ops['Mixup'] = {"alpha": args.mixup, "prob": args.mixup_switch_prob, "epsilon": args.smoothing, "class_num": args.nb_classes}
batch_transform_ops['Cutmix'] = {"alpha": args.cutmix, "prob": args.mixup_switch_prob, "epsilon": args.smoothing, "class_num": args.nb_classes}
mixup_fn = transforms.TransformOpSampler(**batch_transform_ops)

def mixup_collate_fn(batch):
    batch = mixup_fn(batch)
    batch = collate_fn(batch)
    return batch
    
data_loader_train = paddle.io.DataLoader(
    dataset_train, batch_sampler=sampler_train,
    num_workers=8,
    collate_fn=mixup_collate_fn,
)
```
