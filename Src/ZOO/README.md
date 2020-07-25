![](./../../Document/Image/title.png)
# ZOO
## Default Model

The Pytorch / Torchvision implemant a baseline model for different mission 

|Mission|Network(torchvision)|
|---|---|
| "Detection"|models.detection.fasterrcnn_resnet50_fpn|
|"Segmentation"|models.segmentation.deeplabv3_resnet50|
|"InstenceSegmentation"|models.detection.maskrcnn_resnet50_fpn|
|"Classification"|models.resnet50|
|"Keypoint"|models.detection.keypointrcnn_resnet50_fpn|