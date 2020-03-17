# Document with EN_US




## Why Instance ?


![](./quick.jpg)


## Doccument with EN_US











### A typically process：

if we have a set of image & label, we need put the image files & label files into image-folder & label-folder . Now, we have the root path of two folders that include dataset like `./root/image` & `./root/label` . In addition, we will generated `class_name.txt`automaticlly.

Each class name occupies a line (please make sure the file not have gap line).

The detail could get in [DataSet Toolkit](Data/DataToolkit/README.md)

That include all the configurable option about training instance **(not complate)**:


```json
{
    "instance_id": 0,
    "content": {
        "Net": {
            "DefaultNetwork":true,
            "NetType": "DeepLabV3",
            "BatchSize": 2,
            "BackBone": "None",
            "Optimizer":"SGD",
            "Loss_Function":"MSELoss",
            "learning_rate":0.02,
            "momentum":0.9,
            "weight_decay":1e-4,
            "lr_scheduler":"MultiStepLR",
            "lr_steps":[8,11],
            "lr_gamma":0.1,
            "class_num":2

        },
        "Dataset": {
            "DefaultDataset":false,
            "Transform":[
                    {"ToTensor":"None"},
                    {"Normalize":[[0.485,0.456,0.406],[0.229, 0.224, 0.225]]}
                ]
            ,
            "Type": "Costum_NPY_DataSet",
            "DataRatio":0.8,
            "NPY":"/workspace/SampledDatasetMini.npy",
            "root": "/workspace/WSNets/Data/labelme/demo",
            "train_index_file":"annotation.json",
            "val_index_file":"annotation.json"
        
        },
        "Config": {
            "group_factor":0,
            "DistributedDataParallel":false,
            "Resume":false,
            "multiscale_training": true,
            "logdir": "./root/log",
            "devices":"GPU",
            "gpu_id": "0",
            "epochs": 200,
            "down_pretrain_model": false,
            "checkpoint_path": "/workspace/models",
            "visualization": false,
            "worker_num":1

        }
    },
    "MissionType": "Segmentation"
}


```

The Json file content could be modified. When you finish your change , run as :

```bash
python instance.py --cfg instance-{ intance_id }-config.json
```

##### The train will start.

> Pytorch 1.1 has supported for tensorboard run like `tensorboard --logdir ./root/log` to supervise whole training flow. [: Guide to use tensorbord for pytorch](https://pytorch.org/docs/stable/tensorboard.html)









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
****
1.Constructs a type of model with torchvision default model 

    * BackBone-               MNASNetV1.3
    * Detection-              Faster R-CNN model with a ResNet-50-FPN
    * Segmentation-           DeepLabV3 model with a ResNet-50
    * Instence Segmentation-  Mask R-CNN model with a ResNet-50-FPN


With this great model ,we can start different type of mission quickly

2.Constructs a third-party / Custom / SOTA model

Now support On:

    * BackBone-                EfficientNets
    * Detection                YoloV3
    * Segmentation             DeepLab V3 Plus
    * Instence Segmentation    BlendMask
****

You can run the default Mission as :
```python
    instence=Instence(instence_id=0)
    instence.DefaultDetection()
```
Or run the modified model as :

```python
    mymodel=Instence(instence_id=0)
    mymodel.model=MY_Model()# Your Model Class
    mymodel.train()
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



## Reference:
* [EfficientNets](https://arxiv.org/pdf/1905.11946.pdf)

* [MNasNet](https://arxiv.org/pdf/1807.11626.pdf)