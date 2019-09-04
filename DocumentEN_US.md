
![](./quick.jpg)


### **Doc with EN_US**






#### Project Structure
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

MODE=[Segmentation,Detection,InstenceSegmentation]
```

**A typically process：**

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

That include all the configurable option about training instance **(not complate)**:


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
        .......
    }
    }
}

```

The Json file content could be modified. When you finish your change , run as :

```bash
python train.py --cfg instance-(0)-config.json
```

##### The train will start.

> Pytorch 1.1 has supported for tensorboard run like `tensorboard --logdir ./root/log` to supervise whole training flow. 


#### Dataset Generator

##### Data structure rule



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

#### NetworkGenerator

The Network Generator have two way.

1.Constructs a type of model with torchvision default model 

    * BackBone-               MNASNetV1.3
    * Detection-              Faster R-CNN model with a ResNet-50-FPN
    * Segmentation-           DeepLabV3 model with a ResNet-50
    * Instence Segmentation-  Mask R-CNN model with a ResNet-50-FPN


With this great model ,we can start different type of mission quickly

2.Constructs a third-party / a state of arts model

Now support On:


    * BackBone-                EfficientNets
    * Detection                YoloV3
    * Segmentation             --
    * Instence Segmentation    -- 

you can run as :
```python
instence=Instence(instence_id=0)
instence.DefaultDetection()
```
the network will be constructd like:
```bash
---------------- FasterRCNN(
  (transform): GeneralizedRCNNTransform()
  (backbone): BackboneWithFPN(
    (body): IntermediateLayerGetter(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): FrozenBatchNorm2d()
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
      (layer1): Sequential(
        (0): Bottleneck(
          (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d()
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d()
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d()
          (relu): ReLU(inplace=True)
          (downsample): Sequential(
            (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d()
          )
        )
        (1): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d()
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d()
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d()
          (relu): ReLU(inplace=True)
        )
        (2): Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn1): FrozenBatchNorm2d()
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
          (bn2): FrozenBatchNorm2d()
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (bn3): FrozenBatchNorm2d()
          (relu): ReLU(inplace=True)
        )
      )
    
    ........

    (fpn): FeaturePyramidNetwork(
      (inner_blocks): ModuleList(
        (0): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))
        (2): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
        (3): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (layer_blocks): ModuleList(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (extra_blocks): LastLevelMaxPool()
    )
  )
  (rpn): RegionProposalNetwork(
    (anchor_generator): AnchorGenerator()
    (head): RPNHead(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (cls_logits): Conv2d(256, 3, kernel_size=(1, 1), stride=(1, 1))
      (bbox_pred): Conv2d(256, 12, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (roi_heads): RoIHeads(
    (box_roi_pool): MultiScaleRoIAlign()
    (box_head): TwoMLPHead(
      (fc6): Linear(in_features=12544, out_features=1024, bias=True)
      (fc7): Linear(in_features=1024, out_features=1024, bias=True)
    )
    (box_predictor): FastRCNNPredictor(
      (cls_score): Linear(in_features=1024, out_features=91, bias=True)
      (bbox_pred): Linear(in_features=1024, out_features=364, bias=True)
    )
  )
) ---------------
```













