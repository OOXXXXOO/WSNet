![](/Resources/Document/IMG_0932.PNG)
## Network

The Network Module is devide into two individual part：

#### 1.Constructs a type of model with torchvision default model

    * BackBone-                             MNASNetV1.3
    * Detection-                            Faster R-CNN model with a ResNet-50-FPN
    * Segmentation-                         DeepLabV3 model with a ResNet-50
    * Instence Segmentation-                Mask R-CNN model with a ResNet-50-FPN


With these great model ,we can start different type of mission quickly

#### 2.Constructs a third-party / a state of arts model

Now support for:


    * BackBone-                             EfficientNets
    * Detection-                            YoloV3,CascadeRCNN
    * Segmentation-                         DeeplabV3-Xception
    * Instence Segmentation-                Mask R-CNN with DenseNet152 

