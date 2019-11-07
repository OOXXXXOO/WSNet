
![](./quick.jpg)


### **Doccument with EN_US**






#### Project Structure
```python
                 
"""                    
                 |-->Dataset-->|——training-array generator<----->|  
                 |             |——training-to DataLoader         |                  |——template-config-generator——>——>|
                 |                                               |<-->|——Config-----|——readconfig<————————————————————|  
 Instance[MODE]——|                                               |                  |     ^       
                 |                                               |                  |——configure instance—————————————|       
                 |                                               |          
                 |——Network----|——readconfig<———————————————————>|  
                               |——Network Generator
                               |——Network Process——————>|
                                                        |---->Train/Val/Test
"""
    MODE=[Segmentation,Detection,Instence,Caption]
```

#### **A typically process：**

if we have a set of image & label, we need put the image files & label files into image-folder & label-folder . Now, we have the root path of two folders that include dataset like`./root/image` & `./root/label`. In addition, we need to have a text file include class names like `./root/classes.txt` that like:

```
class1
class2
class3
```
Each class name occupies a line (please make sure the file not have gap line).








Its gonna be start a brand new **Instance**,the code will scan the all image file in `./root/image` and try to make a map between the image & label -files . In end of the scan process , program will generate index file of all the mapping files and  default instance configure file 

`./instance-{id}-config.json`

That include all the configurable option about training instance **(not complate)**:


```json
{
    "instance_id": 0,
    "content": {
        "Net": {
            "DefaultNetwork":true,
            "NetType": "FasterRCNN",
            "BatchSize": 4,
            "BackBone": "None",
            "Optimizer":"SGD",
            "Loss_Function":"CrossEntropyLoss"
        },
        "Dataset": {
            "Transform":[
                    "RandomHorizontalFlip",
                    "ToTensor",
                    "Normalize"
                ]
            ,
            "Type": "COCO2014",
            "root": "/media/winshare/98CA9EE0CA9EB9C8/COCO_Dataset",
            "train_index_file":"annotations/instances_train2014.json",
            "val_index_file":"annotations/instances_val2014.json"
        
        },
        "Config": {

            
            "multiscale_training": true,
            "logdir": "./root/log",
            "devices":"GPU",
            "gpu_id": 0,
            "epochs": 200,
            "down_pretrain_model": true,
            "checkpoint_path": "./root/model",
            "visualization": true,
            "worker_num":2

        }
    },
    "MissionType": "Detection"
}


```

The Json file content could be modified. When you finish your change , run as :

```bash
python train.py --cfg instance-{intance_id}-config.json
```

##### The train will start.

> Pytorch 1.1 has supported for tensorboard run like `tensorboard --logdir ./root/log` to supervise whole training flow. [: Guide to use tensorbord for pytorch](https://pytorch.org/docs/stable/tensorboard.html)





****





#### Dataset Generator

##### Data structure rule



For custom dataset， index file & configure neurons network is complex step in training proces. So, that auto generate  & configure is most important part of this project.The network training gonna be **Easy & Quick**.


**Label requiremant**：

**Segmentation**:


The Original image data like 

WSNet/Data/DetectionDatasetDemo/image

```
---Root
    |---image
        |---image1.jpg
        |---image2.jpg
                   `
                   `
                   `
```

The labelTool    -  [labelme](WSNet/Data/labelme/README.md) could be found , you should install labelme with:
```bash
# python3
conda create --name=labelme python=3.6
source activate labelme
# conda install -c conda-forge pyside2
# conda install pyqt
pip install pyqt5  # pyqt5 can be installed via pip on python3
pip install labelme
```
When you finish your label work
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
        |---image1.xml/txt/json
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

Paper Collection

* [EfficientNets](https://arxiv.org/pdf/1905.11946.pdf)

* [MNasNet](https://arxiv.org/pdf/1807.11626.pdf)

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

#### General Transform 
The General_Transform class work for MultiMission Data Transform
    The Pytorch-like Transform just work for image data & different Mission
    have corresponding transform.
      
#####  Detection:

  The models expect a list of Tensor[C, H, W], in the range 0-1. 
  The models internally resize the images so that they have a minimum size of 800. 
  This option can be changed by passing the option min_size to the constructor of the models.

  * boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

  * labels (Int64Tensor[N]): the class label for each ground-truth box



#####  Segmentation:
  
  As with image classification models, all pre-trained models expect input images normalized in the same way. 
  The images have to be loaded in to a range of [0, 1] and then normalized using 

* mean = [0.485, 0.456, 0.406] 

* std = [0.229, 0.224, 0.225]. 

  They have been trained on images resized such that their minimum size is 520.



##### Instance_Segmantation:
  
  During inference, the model requires only the input tensors, and returns the post-processed predictions as a List[Dict[Tensor]], 
  one for each input image. The fields of the Dict are as follows:

  * boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

  * labels (Int64Tensor[N]): the predicted labels for each image

  * scores (Tensor[N]): the scores or each prediction

  * masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range.

  In order to obtain the final segmentation masks, the soft masks can be thresholded, generally with a value of 0.5 (mask >= 0.5)



#####  BackBone:

  The models expect a list of Tensor[C, H, W], in the range 0-1. 



#####  Caption:




#####  KeyPoint:

  During training, the model expects both the input tensors, as well as a targets (list of dictionary), containing:

  * boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

  * labels (Int64Tensor[N]): the class label for each ground-truth box

  * keypoints (FloatTensor[N, K, 3]): the K keypoints location for each of the N instances, in the format [x, y, visibility], where visibility=0 means that the keypoint is not visible.
