> 2019.1029

添加了Optimizer，LossFunction的解析

计划添加target解析器，根据不同的任务类型编写统一的target目标转换,把标签数据转换为供网络解析的target对象，然后根据任务来解析数据输入网络




### Segmentation-Target
The images have to be loaded in to a range of [0, 1] and then normalized using

* mean = [0.485, 0.456, 0.406]

* std = [0.229, 0.224, 0.225] 
 
 They have been trained on images resized such that their minimum size is 520.


### Detection-Target
#### train input
```python
target=
[
    {
        "boxes":boxes,
        "label",label
    },
    {
        "boxes":boxes1,
        "label",label1
    }
]

boxes=(FloatTensor[N, 4])
labels=(Int64Tensor[N])

```
#### test output

```python
output=
[
    {
        "boxes":boxes,
        "label",label
        "scores":scores
    },
    {
        "boxes":boxes1,
        "label",label1
        "scores":scores1
    }
]

boxes=(FloatTensor[N, 4])
labels=(Int64Tensor[N])
scores=(Tensor[N])
```

### Mask-Target
#### train input
```python
target=
[
    {
        "boxes":boxes,
        "labels":label
        "Masks":mask
    },
    {
        "boxes":boxes1,
        "labels":label1
        "Masks":mask1
    }
]

boxes=(FloatTensor[N, 4])
labels=(Int64Tensor[N])
masks=(UInt8Tensor[N, H, W])

```
#### test output

```python
output=
[
    {
        "boxes":boxes,
        "label",label
        "scores":scores
    },
    {
        "boxes":boxes1,
        "label",label1
        "scores":scores1
    }
]

boxes=(FloatTensor[N, 4])
labels=(Int64Tensor[N])
scores=(Tensor[N])
```


### Caption-Target
#### train input
```python
target=
[
    {
        "boxes":boxes,
        "label",label
    },
    {
        "boxes":boxes1,
        "label",label1
    }
]

boxes=(FloatTensor[N, 4])
labels=(Int64Tensor[N])

```
#### test output

```python
output=
[
    {
        "boxes":boxes,
        "label",label
        "scores":scores
    },
    {
        "boxes":boxes1,
        "label",label1
        "scores":scores1
    }
]

boxes=(FloatTensor[N, 4])
labels=(Int64Tensor[N])
scores=(Tensor[N])
```

### KeyPoint-Target
#### train input
```python
target=
[
    {
        "boxes":boxes,
        "label",label
    },
    {
        "boxes":boxes1,
        "label",label1
    }
]

boxes=(FloatTensor[N, 4])
labels=(Int64Tensor[N])

```
#### test output

```python
output=
[
    {
        "boxes":boxes,
        "label",label
        "scores":scores
    },
    {
        "boxes":boxes1,
        "label",label1
        "scores":scores1
    }
]

boxes=(FloatTensor[N, 4])
labels=(Int64Tensor[N])
scores=(Tensor[N])
```
> 2019-10-30

添加了训练流程 基本打通了数据处理

待完成：

Transfrom  应用


```
T.CenterCrop
T.ColorJitter
T.Compose
T.FiveCrop
T.functional.adjust_brightness 
T.functional.adjust_contrast 
T.functional.adjust_gamma 
T.functional.adjust_hue 
T.functional.adjust_saturation 
T.functional.affine 
T.functional.crop 
T.functional.erase 
T.functional.five_crop 
T.functional.hflip 
T.functional.normalize 
T.functional.pad 
T.functional.perspective 
T.functional.resize 
T.functional.resized_crop 
T.functional.rotate 
T.functional.ten_crop 
T.functional.to_grayscale 
T.functional.to_pil_image 
T.functional.to_tensor 
T.functional.vflip 
T.Grayscale
T.Lambda
T.Normalize
T.Pad
T.RandomAffine
T.RandomApply
T.RandomChoice
T.RandomCrop
T.RandomErasing
T.RandomGrayscale
T.RandomHorizontalFlip
T.RandomOrder
T.RandomPerspective
T.RandomResizedCrop
T.RandomRotation
T.RandomSizedCrop
T.RandomVerticalFlip
T.Resize
T.Scale
T.TenCrop
T.ToPILImage
T.ToTensor
```

1031任务
构建Transform来完成输入数据的变换


1107
Transform完成情况：
1.对于图像的Transform以及基本完成
但是诞生了新的问题即官方未指定读入数据的Transform

所以准备构建新的
General Transform类型

1112


一般来说  对于不同任务的图像和标注对应变换  都是最困难的

### Detection

1.图像Resize->boxes坐标 resize
2.图像Flip -> boxes坐标 flip
3.图像rotate -> boxes 坐标 rotate





_____
添加collate_fn

### 问题  搞清楚zip

#### max stack

两种方式的优劣


```python
def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label = zip(*batch)
    pad_label = []
    lens = []
    max_len = len(label[0])
    for i in range(len(label)):
        temp_label = [0] * max_len
        temp_label[:len(label[i])] = label[i]
        pad_label.append(temp_label)
        lens.append(len(label[i]))
    return img, pad_label, lens
```


1114:

终于到达想要的错误
：


torch.Size([3, 427, 640])
torch.Size([3, 640, 426])
torch.Size([3, 436, 640])
torch.Size([3, 334, 500])
torch.Size([3, 640, 505])
torch.Size([3, 360, 640])
torch.Size([3, 425, 640])
Traceback (most recent call last):
  File "Src/general_transfrom.py", line 196, in <module>
    main()
  File "Src/general_transfrom.py", line 186, in main
    for index,(image,target) in enumerate(trainloader):
  File "/home/winshare/anaconda3/envs/stable/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 819, in __next__
    return self._process_data(data)
  File "/home/winshare/anaconda3/envs/stable/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 846, in _process_data
    data.reraise()
  File "/home/winshare/anaconda3/envs/stable/lib/python3.6/site-packages/torch/_utils.py", line 385, in reraise
    raise self.exc_type(msg)
RuntimeError: Caught RuntimeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/home/winshare/anaconda3/envs/stable/lib/python3.6/site-packages/torch/utils/data/_utils/worker.py", line 178, in _worker_loop
    data = fetcher.fetch(index)
  File "/home/winshare/anaconda3/envs/stable/lib/python3.6/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "Src/general_transfrom.py", line 166, in collate_fn
    inp = torch.stack(transposed_data[0], 0)
RuntimeError: invalid argument 0: Sizes of tensors must match except in dimension 0. Got 427 and 436 in dimension 2 at /opt/conda/conda-bld/pytorch_1570910687230/work/aten/src/TH/generic/THTensor.cpp:689



collate_fn最后修正


1115 最终设计


所有的Tranform  在进入collat_fn之前都是 PIL格式

最后在Collate_fn按照所需格式转换为tensor

1118

解决了SGD Self的问题
Optimizer需要在类内实例化

1，开始了第一次训练
---------Epoch: 0
-----Step 0 --LOSS-- tensor(32.7564, device='cuda:0', grad_fn=<AddBackward0>)
-----Step 1 --LOSS-- tensor(21.2340, device='cuda:0', grad_fn=<AddBackward0>)
-----Step 2 --LOSS-- tensor(4.0840, device='cuda:0', grad_fn=<AddBackward0>)
-----Step 3 --LOSS-- tensor(2.3075, device='cuda:0', grad_fn=<AddBackward0>)
-----Step 4 --LOSS-- tensor(0.9804, device='cuda:0', grad_fn=<AddBackward0>)
-----Step 5 --LOSS-- tensor(23.8030, device='cuda:0', grad_fn=<AddBackward0>)
-----Step 6 --LOSS-- tensor(nan, device='cuda:0', grad_fn=<AddBackward0>)
-----Step 7 --LOSS-- tensor(nan, device='cuda:0', grad_fn=<AddBackward0>)
-----Step 8 --LOSS-- tensor(1.0128, device='cuda:0', grad_fn=<AddBackward0>)
-----Step 9 --LOSS-- tensor(20.5972, device='cuda:0', grad_fn=<AddBackward0>)
-----Step 10 --LOSS-- tensor(nan, device='cuda:0', grad_fn=<AddBackward0>)
-----Step 11 --LOSS-- tensor(nan, device='cuda:0', grad_fn=<AddBackward0>)
-----Step 12 --LOSS-- tensor(nan, device='cuda:0', grad_fn=<AddBackward0>)
-----Step 13 --LOSS-- tensor(nan, device='cuda:0', grad_fn=<AddBackward0>)
-----Step 14 --LOSS-- tensor(nan, device='cuda:0', grad_fn=<AddBackward0>)
-----Step 15 --LOSS-- tensor(20.8425, device='cuda:0', grad_fn=<AddBackward0>)
-----Step 16 --LOSS-- tensor(22.1752, device='cuda:0', grad_fn=<AddBackward0>)


待解决问题    

* xmin, ymin, xmax, ymax = boxes.unbind(1)
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)时不时出现的

* eval mode
* nan
* tensor board writer
  [
      Net
      Loss
      Precision
      Image-label
      Image-PredictionBox
  ]

default mode的不同任务适配

* Segmantation Template Json
* Instance Template Json
* KeyPoiny Template Json

Local Logger
seaborn 绘图覆盖

