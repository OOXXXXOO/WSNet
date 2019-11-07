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

```python
class General_Transform():
    def __init__(self):
        



```





