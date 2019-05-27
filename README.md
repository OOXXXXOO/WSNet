![](./timg.jpeg)


## WSNet Document
> Version 1.0

### The primary feature of WSNet toolkit in **version 1.0**:

* The custom data structure not has a standard form,so this toolkit rule a easy way to make that support for different NeuroNetwork ( For Detection , Segmentation or both like Mask RCNN )

* Detection Support for :
  * Twostage（Faster RCNN ,Cascade RCNN ,Mask RCNN）;
  * Onestage （YOLO v3 ,SSD）;

* Segmentation Support for : 
    * Deeplab v3+，Cascade Net，Seg Net

>  In principle , When we run 'datasetbuild' will generate the dataset and relate configure file , if we custom our data with
>  **Data structure RULE.**
>
>  You could modified the configure file to change option like gpu-id ，NetWork ...... or download pre-train model( Actually That alway is a efficient way to boost your training work). 

****




## **Doc with EN_US**






### Project Structure
```txt
                              |——template-config-generator——>——>|
                |——Config-----|——readconfig<————————————————————|  
                |     ^       |——configure instance             |  
                |     |                                         |               
Instance[MODE]——|-->Dataset-->|——training-array generator       |  
                |             |——training-to DataLoader         |  
                |                                               |          
                |——Network----|——readconfig<————————————————————|  
                              |——Network Generator
                              |——Network Process——————>——————————>Train/Val/Test

MODE=[Segmentation,Detection,Mask]
```


A typically process：

if we have a set of image & label, we need put the image files & label files into image-folder & label-folder . Now, we have the root path of two folders that include dataset like`./root/image` & `./root/label`. In addition, we need to have a text file include class names like `./root/classes.txt` that like:

```
class1
class2
class3
```
Each class name occupies a line (please make sure the file not have gap line).

Then , run as below:
```bash
python datasetbuilder.py --root ./root  --mode segmentation
```
Its gonna be start a brand new **generate instance**,the code will scan the all image file in `./root/image` and try to make a map between the image & label -files . In end of the scan process , program will generate index file of all the mapping files and  default instance configure file 

`./instance-(id)-config.json`

That include all the configurable option about training instance:


```json
{
    "instance_id":0,
    "mode":0,
    "content":{
        "Net":{
            "BackBone":"None",
            "NetType":"DeeplabV3plus",
            "BatchSize":4
        },
    "Dataset":{
        "root":"./root"
    },
    "Config":{
        "gpu_id":0,
        "epochs":100,
        "down_pretrain_model":true,
        "checkpoint_path":"./root/model",
        "mutilscale_training":true,
        "logdir":"./root/log"
    }
    }
}

```

The Json file content could be modified. When you finish your change , run as :

```bash
python train.py --cfg config.json
```

#### The train will start.

> Pytorch 1.1 has supported for tensorboard run like `tensorboard --logdir ./root/log` to supervise whole training flow. 


## Dataset Generator

### Data structure rule



For custom dataset， index file & configure neurons network is complex step in training proces. So, that auto generate  & configure is most important part of this project.The network training gonna be **Easy & Quick**.


**Label requiremant**：

**Segmentation**:
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


> * The image file of label is pixel mask label, the ascending sort sequence of different pixel values map on classnames.
>
> * if save the class names by txt file, the individual classname occupies a single line and please make sure without gap line . if csv file, split class names by char `,`
>
> * The **Sequence** generate by auto scanning, so we recommand label file is single channel image. Btw, the 3ch image also could be support. The difference is visualization method。 


**Detection**:
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





> * Usually, label tools is labelme or labelimg, so the label file could be xml/json/bbox file.
> * Current version support for json & bbox file 



****



# Doc with ZH_CN


### 1.0 阶段该工具主要功能:


* 该工具基于Pytorch 在第一版中完成了对于Detection，Segmentation以及Mask（RCNN）相关任务的自定义数据集规范，以及其对应的数据集生成器。

* 对RCNN系列网络的支持（Faster RCNN 、Cascade RCNN 、Mask RCNN）、对Onestage 的YOLO v3 ,SSD
  的支持

* 对Deeplab v3+, CascadeNet SegNet的支持


> 理论上，以上完成的功能，在按照
> **数据组织规则**
> 规范组织数据后，执行 'Datasetbuide' 既能够生成数据集，以及相关的配置文件。修改配置文件中的可选选项之后即可展开快速训练（如果不需要预训练模型的话，可以不这么做,但这么做通常是有效的）。

****






### 整体结构
```txt


                      |——ReadConfig
                      |——Config2Net
          |——Net------|——Train/val Process flow
          |
Instance——|——DataSet--|——datasetbuild->(data，label)
          |
          |——Config---|——Mission type ( segmentation/detection/mask )
                      |——Net(BackBone)
                      |——Optimizer
                      |——Super parameters
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
    "instance_id":0,
    "mode":0,
    "content":{
        "Net":{
            "BackBone":"None",
            "NetType":"DeeplabV3plus",
            "BatchSize":4
        },
    "Dataset":{
        "root":"./root"
    },
    "Config":{
        "gpu_id":0,
        "epochs":100,
        "down_pretrain_model":true,
        "checkpoint_path":"./root/model",
        "mutilscale_training":true,
        "logdir":"./root/log"
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
> 3, 数据生成过程中对于一类任务的像素类别会自动扫描，所以推荐label为单通道，但是三通道也同样支持，只不过在可视化过程中会有不同策略



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
> 当前版本支持json以及bbox格式
> 




### ROI Pooling

### ROI Align

### RPN 网络


















## DataSetBuilder:

### Data Structure Rule

### Segmentation Dataset


### Detection Dataset


### Mask Dataset

## ROI Pooling

## RPN Network


****
### **MIT License**
#### Copyright (c) 2018 The Python Packaging Authority

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

**The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.**

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.