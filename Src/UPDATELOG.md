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
