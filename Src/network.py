# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    network.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/28 11:46:19 by winshare          #+#    #+#              #
#    Updated: 2020/02/28 11:50:20 by winshare         ###   ########.fr        #
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

import torch
import torch.utils.model_zoo
import torchvision.models as models
import torch.nn as nn
import random

class NETWORK(CFG):
    def __init__(self):
        CFG.__init__(self)
        """
            The Network Generator have two way.
            1.Constructs a type of model with torchvision default model 
                * BackBone-               MNASNetV1.3
                * Detection-              Faster R-CNN model with a ResNet-50-FPN
                * Segmentation-           DeepLabV3 model with a ResNet-50
                * Instence Segmentation-  Mask R-CNN model with a ResNet-50-FPN
                * KeyPoint-               KeyPointRCNN model with a ResNet-50-FPN

            With this great model ,we can start different type of mission quickly
            
            2.Constructs a third-party / a state of arts model
            Now support On:
                * BackBone-                EfficientNets
                * Detection                YoloV3
                * Segmentation             --
                * Instence Segmentation    -- 
        """

        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #




        self.debug=True
        


        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #

        # ---------------------------------------------------------------------------- #
        #                            Network Dictionary Init                           #
        # ---------------------------------------------------------------------------- #
        self.modeldict={
            "Detection":self.DefaultDetection,
            "Segmentation":self.DefaultSegmentation,
            "BackBone":self.DefaultBackBone,
            "Instence Segmentation":self.DefaultInstenceSegmentation,
            "KeyPoint":self.DefaultKeyPoint,
        }
        self.model=None
        # ---------------------------------------------------------------------------- #
        #                       Optimizer init & Instantialation in DefaultNetwork     #
        # ---------------------------------------------------------------------------- #

        if self.DefaultNetwork:
            print('\n\n-----Use The Default Network')
            self.modeldict[self.MissionType](pretrained=self.download_pretrain_model)
            # print(self.model)
            
        # --------------------------------- optimizer -------------------------------- #
            print('\n\n--------------------------- Network General Info: --------------------------  ')
            self.optimizer=self.optimizer(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay
            )
            print('Network optimizer:',self.optimizer)

        
        # ------------------------------- lr_scheduler ------------------------------- #

            self.lr_scheduler=self.lr_scheduler(
                self.optimizer,
                milestones=self.lr_steps,
                gamma=self.lr_gamma
            )
            print('Network lr_scheduler:',self.lr_scheduler)
            print('--------------------------- Network General Info: --------------------------')
        print("\n\n----------------------- NETWORK Class Init Successful ----------------------\n\n")







        # ---------------------------------------------------------------------------- #
        #                             DefaultNetwork Option                            #
        # ---------------------------------------------------------------------------- #


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
        if self.debug:
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
        if self.debug:
            print('\n\n----------------',self.model,'---------------\n\n')

    def NetWorkInfo(self):
        print('Device info - GPU CUDA useful : ',torch.cuda.is_available())
        if torch.cuda.is_available():
            print('\t\t|==========>GPU Count',torch.cuda.device_count(),'\n\n')

    
    # ----------------------------- Collate Function for Temperory ----------------------------- #

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets
