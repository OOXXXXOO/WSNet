# WSNet Document
> Version 1.0

### 1.0阶段该工具主要功能:

* 该工具基于Pytorch 在第一版中完成了对于Detection，Segmentation以及Mask（RCNN）相关任务的自定义数据集规范，以及其对应的数据集生成器。

* 对RCNN系列网络的支持（Faster RCNN 、Cascade RCNN 、Mask RCNN）、对Onestage 的YOLO v3 ,SSD
  的支持

* 对Deeplab v3+, Cascade Net SegNet的支持


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

# Doc with ZH_CN

## 数据集生成器

### 数据组织规范

对于常见的数据整理形式来说，索引文件的建立以及配置相关神经网络的步骤通常是最复杂的，所以对这个步骤的自动化处理就变成了非常重要的部分。




#### 分割数据集


#### 检测数据集


#### Mask数据集


### 分割数据集生成

### 检测数据集生成

### Mask数据集生成

## ROI Pooling

## RPN 网络



# Doc with EN_US

## DataSetBuilder:

### Data Structure Rule

### Segmentation Dataset


### Detection Dataset


### Mask Dataset

## ROI Pooling

## RPN 网络