# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    transform.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/28 11:46:45 by winshare          #+#    #+#              #
#    Updated: 2020/03/20 17:40:35 by winshare         ###   ########.fr        #
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




import sys
import os
sys.path.append(sys.path[0][:-19])
print(sys.path)
import glob
import PIL.Image as Image
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import torchvision.datasets as dataset
import pycocotools.mask as mask
# ---------------------------- official reference ---------------------------- #

import matplotlib.pyplot as plt

# ------------------------------ local reference ----------------------------- #


import Src.Utils.Transform.box.data_aug as BT
import Src.Utils.Transform.mask.segmentation_transforms as MT

# ------------------------ Pytorch Official Functional Transform  ------------------------ #





segmentation_transform={
"Normalize":MT.Normalize
"ToTensor":MT.ToTensor
"RandomHorizontalFlip":MT.RandomHorizontalFlip
"RandomRotate":MT.RandomRotate
"RandomGaussianBlur":MT.RandomGaussianBlur
"RandomScaleCrop":MT.RandomScaleCrop
"FixScaleCrop":MT.FixScaleCrop
"FixedResize":MT.FixedResize
}

detection_transform={
"RandomHorizontalFlip":A.RandomHorizontalFlip,
"HorizontalFlip":A.HorizontalFlip,
"RandomScale":A.RandomScale,
"Scale":A.Scale,
"RandomTranslate":A.RandomTranslate,
"Translate":A.Translate,
"RandomRotate":A.RandomRotate,
"Rotate":A.Rotate,
"RandomShear":A.RandomShear,
"Shear":A.Shear,
"Resize":A.Resize,
"RandomHSV":A.RandomHSV 
}




class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self,data):
        for t in self.transforms:
            data = t(data)
        return data





class GeneralTransform():
    def __init__(self,transformlist=None,Mission=None):
        """
        The official dataset getitem function dont have anyway to make transform support
        for all of them .
        So the General Transform class is a resolution :
        
        Init General Transform 
        
        """
        SupportMission=[
            "Detection",
            "Segmentation",
            "InstanceSegmentation",
            "KeyPoint",
            "Caption"
        ]
        print("----------------------------- GeneralTransform -----------------------------")
        assert not transformlist==None,"Invalid General Transform List"
        assert Mission in SupportMission,"Invalid Transform Dictionary"
        self.Mission=Mission



        # -------------- Filte The Transform String list is valid or not ------------- #

        if self.Mission=="Detection" or self.Mission=="InstanceSegmentation":
            """
            Dict Data Only
            """
            pass
        if self.Mission=="Segmentation":
            """
            Ndarray Mask Only 
            """
            pass
        
        print("\n\n-----Transform Init with Mode",self.Mission,"-----")
        
        for process in self.Target_SupprtDict.keys():
            print('-----support :',process)
            
        print("-----Transform Init with Mode",self.Mission,"-----\n\n")
        
    

    def maskTransform(self,target):
        pass



    def box_ndarraytransform(self,target):
        pass



    def imagery_transform(self,image):
        pass
        



    def __call__(self,data):
        """
        data must be dict
        "image":
            ndarray or PIL image
        "target":
            dict={
                - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
                        between 0 and H and 0 and W
                - labels (Int64Tensor[N]): the class label for each ground-truth box
                - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance
            }
        """
        IM=data['image']
        GT=data['target']
        if "boxes" in GT.keys():
        
        if "labels" in GT.keys():

        if "mask" in GT.keys():
















def main():

    # ---------------------- Test Part for GeneralTransform ---------------------- #

    print("\n\n---------------------- Test Part for GeneralTransform ----------------------")
    
    # ---------------------------------- Usage: ---------------------------------- #
    
    DemoTransformDict=[
        # {"RandomResizedCrop":512},  
        # {"RandomRotation":90},
        {"RandomCrop":"None"},
        {"ToTensor":"None"},
        {"Normalize":[[0.485,0.456,0.406],[0.229, 0.224, 0.225]]},
    ]

    

    # Detection:
    datasets=dataset.CocoDetection("/workspace/WSNets/Data/datasets/labelme/demo/train2014/",
    "./Data/datatoolkit/dataset/annotation.json")
    data=datasets[0]
    print(data)
    
    A=GeneralTransform(DemoTransformDict,"InstanceSegmentation")



    
    
    # Instance:


    
    
    # Keypoint:







if __name__ == '__main__':
    main()
     