## WSNet Document
> Version 1.0

### 1.0 阶段该工具主要功能:

* 该工具基于Pytorch 在第一版中完成了对于Detection，Segmentation以及Mask（RCNN）相关任务的自定义数据集规范，以及其对应的数据集生成器。

* 对RCNN系列网络的支持（Faster RCNN 、Cascade RCNN 、Mask RCNN）、对Onestage 的YOLO v3 ,SSD
  的支持

* 对Deeplab v3+, CascadeNet SegNet的支持


> 理论上，以上完成的功能，在按照
> **数据组织规则**
> 规范组织数据后，执行 'Datasetbuide' 既能够生成数据集，以及相关的配置文件。修改配置文件中的可选选项之后即可展开快速训练（如果不需要预训练模型的话，可以不这么做,但这么做通常是有效的）。

****

### The main function of toolkit in version 1.0:

* The custom data structure not has a standard form,so this toolkit rule a easy way to make that support for different NeuroNetwork(For Detection , Segmentation or both like Mask RCNN)

* Support for Twostage（Faster RCNN 、Cascade RCNN 、Mask RCNN），Onestage （YOLO v3 ,SSD）

* Support for Deeplab v3+，Cascade Net，Seg Net

> In principle,When we run 'datasetbuild' will generate the dataset and relate configure file , if we custom our data with
>  **RULE**.
>  You could modified the configure file to change option like gpu-id ，NetWork ...... or download pre-train model( Actually That alway is a efficient way to boost your training work). 

****

# Doc with ZH_CN


### 整体结构
```


                      |——ReadConfig
                      |——Config2Net
          |——Net------|——Train/val Process flow
          |
Instance——|——DataSet--|——datasetbuild->(data，label)
          |
          |——Config---|——Mission type ( segmentation/detection/mask )
                      |——Net(BackBone)
                      |——Optimizer
                      |——Super parameter
                      |——gpu id(set -1 for disable gpu)
                      |——...


```
一个典型的处理流程，如现在获取了一批标注完成的数据，且label和imagefile按照所要求的目录结构已经放置好，例如当前数据存储于`./root/image/`,标注文件存储于`./root/label/`.类名文件`classes.txt`中包含如
```
class1
class2
class3
```
-类别名称信息（如同前面提示的 类名的顺序，必须和标注像素值的大小顺序一致）

```bash
python datasetbuilder.py --root ./root --datasetdir ./root/dataset --mode segmentation
```
执行，会自动扫描`./root/image/`中的图片文件，尝试建立与label文件夹中label文件的索引（也就意味着无论两个文件夹中的文件是否对等，只取能够建立索引关系的文件建立数据集），当所有文件扫描完毕之后，生成训练列表，以及默认配置文件-
`config.json`

```json
{
    instance id:0,
    mode:segmentation,
    content:
    {
        Net:{
            backbone:"None",
            NetType:"Deeplabv3",
            BatchSize:4,

        },
        Dataset:{
            root:"./root"
        
        },
        Config:{
            gpu_id:0,
            epochs:100,
            download_pretrain_model:True,
            checkpointdir:"./root/model",
            mutilscale_training:True,
            logdir:
        }

    }
}

```
当自定义完相应的属性之后保存config.json
运行
```bash
python train.py --cfg config.json
```
即可开始训练,Pytorch 1.1原生支持了tensorboard 可以运行
tensorboard --logdir 监控训练

## 数据集生成器

### 数据组织规范

对于常见的数据整理形式来说，索引文件的建立以及配置相关神经网络的步骤通常是最复杂的，所以对这个步骤的自动化处理就变成了非常重要的部分。因此本工具制定了一套简单的数据处理规范，以及自动化生成配置文件，的快速训练流程。
对于常见的AI训练任务能够快速的生成数据集，训练配置文件。快速开始训练。



**标注要求**：

**分割**:
```
---Root
    |---image
        |---image1.jpg
                   `
                   `
                   `
    |---label
        |---image1.png
                    `
                    `
                    `
    |---classes.txt/csv

```
> 1，注意 label 中的图片为像素级标注的图片文件，色值为从小到大排序,这个顺序对应类别文件（classes.txt）中的类名的顺序。
> 
> 2，如果以txt保存，则一类的名字为一行，如果以csv保存，则类名之间用','分隔。
>




**检测**:
```
---Root
    |---image
        |---image1.jpg
                   `
                   `
                   `
    |---label
        |---image1.xml/txt
                    `
                    `
                    `
    |---classes.txt/csv

```
> 对于检测的标注一般使用labelme，image生成labelimag.xml/json文件，或者bbox.txt文件
> 
```


```





### ROI Pooling

### ROI Align

### RPN 网络







# Doc with EN_US

## DataSetBuilder:

### Data Structure Rule

### Segmentation Dataset


### Detection Dataset


### Mask Dataset

## ROI Pooling

## RPN Network


****

#### Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.