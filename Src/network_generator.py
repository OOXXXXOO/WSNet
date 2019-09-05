import torch
import torch.utils.model_zoo
import torchvision.models as models


class NetworkGenerator():

    def __init__(self):
        

        print('\n\n-----Neural Network Class Init-----\n\n')
        self.model=None




    """
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
    """


    def DefaultGreatBackBone(self,pretrained=False, progress=True,):
        """
        MNASNet with depth multiplier of 1.3 from “MnasNet: Platform-Aware Neural Architecture Search for Mobile”. 
        :param 
        pretrained: If True, returns a model pre-trained on ImageNet 
        :type pretrained: bool 
        :param 
        progress: If True, displays a progress bar of the download to stderr 
        :type progress: bool
        """
        self.model=models.mnasnet1_3(pretrained=pretrained, progress=progress, **kwargs)

    def DefaultDetection(self,pretrained=False, progress=True, num_classes=91, pretrained_backbone=False, **kwargs):
        """
        Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.


        The models expect a list of Tensor[C, H, W], in the range 0-1. 
        The models internally resize the images so that they have a minimum size of 800. 
        This option can be changed by passing the option min_size to the constructor of the models.

            boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

            labels (Int64Tensor[N]): the class label for each ground-truth box
        
        """
        self.model=models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=progress, num_classes=num_classes, pretrained_backbone=pretrained_backbone, **kwargs)
        print('\n\n----------------',self.model,'---------------\n\n')


    def DefaultSegmentation(self,pretrained=False, progress=True, num_classes=21, aux_loss=None):
        """
        Constructs a DeepLabV3 model with a ResNet-50 backbone.

        As with image classification models, all pre-trained models expect input images normalized in the same way. 
        The images have to be loaded in to a range of [0, 1] and then normalized using 
           mean = [0.485, 0.456, 0.406] 
           std = [0.229, 0.224, 0.225]. 
        They have been trained on images resized such that their minimum size is 520.
        """

        self.model=models.segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=progress, num_classes=num_classes, aux_loss=aux_loss)
        print('\n\n----------------',self.model,'---------------\n\n')


    def DefaultInstenceSegmentation(self,pretrained=False, progress=True, num_classes=91, pretrained_backbone=True, **kwargs):
        """
        Constructs a Mask R-CNN model with a ResNet-50-FPN backbone.


        During inference, the model requires only the input tensors, and returns the post-processed predictions as a List[Dict[Tensor]], 
        one for each input image. The fields of the Dict are as follows:

            boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

            labels (Int64Tensor[N]): the predicted labels for each image

            scores (Tensor[N]): the scores or each prediction

            masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range.
       
         In order to obtain the final segmentation masks, the soft masks can be thresholded, generally with a value of 0.5 (mask >= 0.5)
        """
        self.model=models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained, progress=progress, num_classes=num_classes, pretrained_backbone=pretrained_backbone, **kwargs)
        print('\n\n----------------',self.model,'---------------\n\n')

    def NetWorkInfo(self):
        print('Device info - GPU CUDA useful : ',torch.cuda.is_available())
        if torch.cuda.is_available():
            print('\t\t|==========>GPU Count',torch.cuda.device_count(),'\n\n')
    