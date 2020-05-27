# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    network.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/28 11:46:19 by winshare          #+#    #+#              #
#    Updated: 2020/05/27 12:19:39 by winshare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# Copyright 2020 winshare
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from config import CFG
from Src.Nets.BackBone.efficientnet.model import EfficientNet

# ------------------------------ local reference ----------------------------- #

import torch
import torch.utils.model_zoo as zoo
import torchvision.models as models
import torch.nn as nn
import random

# ---------------------------- official reference ---------------------------- #

class NETWORK(CFG):
    def __init__(self):
        CFG.__init__(self)
        """
        1.Constructs a type of model with torchvision default model

            * BackBone-                             MNASNetV1.3
            * Detection-                            Faster R-CNN model with a ResNet-50-FPN
            * Segmentation-                         DeepLabV3 model with a ResNet-50
            * Instence Segmentation-                Mask R-CNN model with a ResNet-50-FPN


        With these great model ,we can start different type of mission quickly

        2.Constructs a third-party / a state of arts model

        Now support for:
            * BackBone-                             EfficientNets
            * Detection-                            YoloV3
            * Segmentation-                         DeeplabV3-Xception
            * Instence Segmentation-                Mask R-CNN with DenseNet152 
        """

        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #




        self.debug=False
        


        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #

        # ---------------------------------------------------------------------------- #
        #                            Network Dictionary Init                           #
        # ---------------------------------------------------------------------------- #
        self.default_modeldict={
            "Detection":self.DefaultDetection,
            "Segmentation":self.DefaultSegmentation,
            "BackBone":self.DefaultBackBone,
            "InstenceSegmentation":self.DefaultInstenceSegmentation,
            "KeyPoint":self.DefaultKeyPoint,
        }
        self.custom_modeldict={
            "BackBone":EfficientNet
            

            # ----------------------------------- Yolo3 ---------------------------------- #

            # ------------------------------- MaskRCNN-FPN ------------------------------- #

            # ----------------------- DeeplabV3+ Xception Need Add ----------------------- #
                        
        }

        self.model=None
        # ---------------------------------------------------------------------------- #
        #                       Optimizer init & Instantialation in DefaultNetwork     #
        # ---------------------------------------------------------------------------- #

        if self.DefaultNetwork:
            print('\n\n-----Use The Default Network')
            self.default_modeldict[self.MissionType](pretrained=self.download_pretrain_model)
   
            
        # --------------------------------- optimizer -------------------------------- #

            print("# ---------------------------- Optimizer&Scheduler --------------------------- #")

            self.optimizer=self.optimizer(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
            )
            print('-----Network optimizer:\n',self.optimizer)

        
        # ------------------------------- lr_scheduler ------------------------------- #

            self.lr_scheduler=self.lr_scheduler(
                self.optimizer,
                milestones=self.lr_steps,
                gamma=self.lr_gamma
            )
            print('-----Network lr_scheduler:\n',self.lr_scheduler)
                
        print("# ---------------------------------------------------------------------------- #")
        print("#                         NETWORK Class Init Successful                        #")
        print("# ---------------------------------------------------------------------------- #")



        if self.download_pretrain_model:
            self._initialize_weights(self.model)



        # ---------------------------------------------------------------------------- #
        #                             DefaultNetwork Option                            #
        # ---------------------------------------------------------------------------- #

    def _initialize_weights(self,model):
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def DefaultKeyPoint(self,pretrained=False, progress=True):
        """
        During training, the model expects both the input tensors, as well as a targets (list of dictionary), containing:

        boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

        labels (Int64Tensor[N]): the class label for each ground-truth box

        keypoints (FloatTensor[N, K, 3]): the K keypoints location for each of the N instances, in the format [x, y, visibility], where visibility=0 means that the keypoint is not visible.
        
        """
        self.model=models.detection.keypointrcnn_resnet50_fpn(pretrained=pretrained, progress=progress, num_classes=2, num_keypoints=17, pretrained_backbone=True)


    def DefaultBackBone(self,pretrained=False, progress=True):
        """
        MNASNet with depth multiplier of 1.3 from “MnasNet: Platform-Aware Neural Architecture Search for Mobile”. 
        :param 
        pretrained: If True, returns a model pre-trained on ImageNet 
        :type pretrained: bool 
        :param 
        progress: If True, displays a progress bar of the download to stderr 
        :type progress: bool
        """
        self.model=models.mnasnet1_3(pretrained=pretrained, progress=progress)

    def DefaultDetection(self,pretrained=True, progress=True, num_classes=91, pretrained_backbone=True):
        """
        Constructs a Faster R-CNN model with a ResNet-50-FPN backbone.


        The models expect a list of Tensor[C, H, W], in the range 0-1. 
        The models internally resize the images so that they have a minimum size of 800. 
        This option can be changed by passing the option min_size to the constructor of the models.

            boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

            labels (Int64Tensor[N]): the class label for each ground-truth box
        
        """
        self.model=models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained, progress=progress, num_classes=num_classes, pretrained_backbone=pretrained_backbone)
        if self.debug:
            print('\n\n----------------',self.model,'---------------\n\n')


    def DefaultSegmentation(self,pretrained=False, progress=True, num_classes=2, aux_loss=None):
        """
        Constructs a DeepLabV3 model with a ResNet-50 backbone.

        As with image classification models, all pre-trained models expect input images normalized in the same way. 
        The images have to be loaded in to a range of [0, 1] and then normalized using 
           mean = [0.485, 0.456, 0.406] 
           std = [0.229, 0.224, 0.225]. 
        They have been trained on images resized such that their minimum size is 520.
        """

        self.model=models.segmentation.deeplabv3_resnet101(pretrained=pretrained, progress=progress, num_classes=num_classes, aux_loss=aux_loss)
        if self.debug:
            print('\n\n----------------',self.model,'---------------\n\n')


    def DefaultInstenceSegmentation(self,pretrained=False, progress=True, num_classes=91, pretrained_backbone=True, **kwargs):
        """
        Constructs a Mask R-CNN model with a ResNet-50-FPN backbone.


        During training, the model expects both the input tensors, as well as a targets (list of dictionary),
        containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
          between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance
        
        """
        self.model=models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained, progress=progress, num_classes=num_classes, pretrained_backbone=pretrained_backbone, **kwargs)
        if self.debug:
            print('\n\n----------------',self.model,'---------------\n\n')

    def NetWorkInfo(self):
        print('Device info - GPU CUDA useful : ',torch.cuda.is_available())
        if torch.cuda.is_available():
            print('\t\t|==========>GPU Count',torch.cuda.device_count(),'\n\n')

    




def main():
    input=torch.randn((2,3,768,768))
    model=models.segmentation.deeplabv3_resnet101(pretrained=False,num_classes=2)
    model.to("cuda")
    from Src.Utils.Summary.Analysis import summary
    summary(model,(1,3,512,512))
    # output=model(input)
    # print(output.size())

if __name__ == '__main__':
    main()
    

