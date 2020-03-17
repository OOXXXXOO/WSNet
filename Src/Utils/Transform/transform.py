# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    transform.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/28 11:46:45 by winshare          #+#    #+#              #
#    Updated: 2020/03/17 16:44:08 by winshare         ###   ########.fr        #
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
# ---------------------------- official reference ---------------------------- #

import matplotlib.pyplot as plt

# ------------------------------ local reference ----------------------------- #

from Src.Utils.Transform.custom import *
import Src.Utils.Transform.data_aug.data_aug as A


# ------------------------ Pytorch Official Functional Transform  ------------------------ #

Functional={
"adjust_brightness":T.functional.adjust_brightness,
"adjust_contrast":T.functional.adjust_contrast,
"adjust_gamma":T.functional.adjust_gamma,
"adjust_hue":T.functional.adjust_hue,
"adjust_saturation":T.functional.adjust_saturation,
"affine":T.functional.affine,
"crop":T.functional.crop,
"erase":T.functional.erase,
"five_crop":T.functional.five_crop,
"hflip":T.functional.hflip,
"normalize":T.functional.normalize,
"pad":T.functional.pad,
"perspective":T.functional.perspective,
"resize":T.functional.resize,
"resized_crop":T.functional.resized_crop,
"rotate":T.functional.rotate,
"ten_crop":T.functional.ten_crop,
"to_grayscale":T.functional.to_grayscale,
"to_pil_image":T.functional.to_pil_image,
"to_tensor":T.functional.to_tensor,
"vflip":T.functional.vflip
}

# ------------------------- Official Transform Class ------------------------- #

ClassFunction={
"Grayscale":T.Grayscale,
"Lambda":T.Lambda,
"Normalize":T.Normalize,
"Pad":T.Pad,
"RandomAffine":T.RandomAffine,
"RandomApply":T.RandomApply,
"RandomChoice":T.RandomChoice,
"RandomCrop":T.RandomCrop,
"RandomErasing":T.RandomErasing,
"RandomGrayscale":T.RandomGrayscale,
"RandomHorizontalFlip":T.RandomHorizontalFlip,
"RandomOrder":T.RandomOrder,
"RandomPerspective":T.RandomPerspective,
"RandomResizedCrop":T.RandomResizedCrop,
"RandomRotation":T.RandomRotation,
"RandomSizedCrop":T.RandomSizedCrop,
"RandomVerticalFlip":T.RandomVerticalFlip,
"Resize":T.Resize,
"Scale":T.Scale,
"TenCrop":T.TenCrop,
"ToPILImage":T.ToPILImage,
"ToTensor":T.ToTensor,
}

# ------------------------ Transform With Pixel Value ------------------------ #
WithValue={
"Normalize":T.Normalize,
"Normalize":T.Normalize,
"ToPILImage":T.ToPILImage,
"ToTensor":T.ToTensor,
"RandomGrayscale":T.RandomGrayscale
}

# ------------------ NeedPara Transform Without Pixel Value ------------------ #

NeedsPara={
"Pad":T.Pad,#填充
"Resize":T.Resize,
"Scale":T.Scale,
"TenCrop":T.TenCrop
}

# -------------------------- Random Transform Without Pixel Value ------------------------- #

Random={
"RandomAffine":T.RandomAffine,#随机仿射变换
"RandomCrop":T.RandomCrop,#随机自由裁切
"RandomErasing":T.RandomErasing,#随机擦除
"RandomGrayscale":T.RandomGrayscale,#随机灰度
"RandomHorizontalFlip":T.RandomHorizontalFlip,#随机水平翻转
"RandomPerspective":T.RandomPerspective,#随机透视变换
"RandomResizedCrop":T.RandomResizedCrop,#随机重采样并裁切
"RandomRotation":T.RandomRotation,#随机旋转
"RandomVerticalFlip":T.RandomVerticalFlip,#随机垂直翻转
}


# ----------------------------- Custom Transform ----------------------------- #

Detection_Overall={
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
        Args:
            
            transform: list of transform dict like:
            
            ```json
            [
                {"Normalize":[[0.485,0.456,0.406],[0.229, 0.224, 0.225]]},
                {"RandomSizedCrop":512},
                {"RandomRotation":90},
                {"ToTensor":"None"}
            ]
            ```
            SupportMission=[
                "Detection",
                "Segmentation",
                "InstanceSegmentation",
                "KeyPoint",
                "Caption"
            ]

        __call__:(image,target)
        image  :
            ndarray,PIL image
        target :
            dict={
                "boxes":
                    [[ 53.         68.0000175 405.        478.9998225   0.       ]
                    [202.         20.99992   496.        486.99978     0.       ]
                    [589.         77.0001275 737.        335.9999825   0.       ]
                    [723.        327.000125  793.        396.000295    1.       ]],
                "class index":
                    [1,3,14,4],
                "segmantation":
                    [[poinset1],[pointset2],[pointset3],[pointset4]]        
                }
            ndarray:
                ndarray,PIL image
        
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


        if self.Mission=="Detection" or self.Mission=="InstanceSegmentation"::
            """
            {
                boxes:      list of box tensor[n,4]                 (float32)
                masks:      list of segmentation mask points [n,n]  (float32)
                keypoints： list of key pointss[n,n]                (float32)
                labels:     list of index of label[n]               (int64)
            }
            """
            self.supprtlist=Detection_Overall
            self.supprtlist["ToPILImage"]=T.ToPILImage
            self.supprtlist["ToTensor"]=T.ToTensor
        
        if self.Mission==""
        
        
        
        
        
        
        
        for transform in transformlist:
            # print(transform)
            for transform_key,transform_para in transform.items():
                if transform_key in ClassFunction or transform_key in Detection_Overall:
                    print("-----Valid Transform : |",transform_key," | with Para: |",transform_para,"|")
                else:
                    print("-----Invalid Transform :",transform)
    

    def Target_dictTransform(self,target):


    def Target_ndarraytransform(self,target):

    def Imagery_transform(self,image):
        

    def filter(self,transfrom_name):
        """
        devide transform into 
            if dict training data:
                image transform
                dict detection transform
                    |-draw mask on background from dict
                        |-do the transform to mask
                    |-do the boxes transform 
            if ndarray (mask) data:
                

        """
        if self.Mission=="Detection":
            self.




    def __call__(self,images,targets):
        if isinstance(targets,dict):
            images=self.Imagery_transform(images)
            targets=self.Target_dictTransform(targets)
        else:
            images=self.Imagery_transform(images)
            targets=self.Target_ndarraytransform(targets)
        return images,targets













def main():

    # ---------------------- Test Part for GeneralTransform ---------------------- #

    print("\n\n---------------------- Test Part for GeneralTransform ----------------------")
    
    # ---------------------------------- Usage: ---------------------------------- #
    
    DemoTransformDict=[
        {"RandomResizedCrop":512},
        {"RandomRotation":90},
        {"ToTensor":"None"},
        {"Normalize":[[0.485,0.456,0.406],[0.229, 0.224, 0.225]]},
    ]

    

    # Detection:
    datasets=dataset.CocoDetection("/workspace/WSNets/Data/labelme/demo/train2014","Src/Utils/DataToolkit/annotation.json")
    for i in range(len(datasets)):
        data=datasets[i]
    A=GeneralTransform(DemoTransformDict,"Detection")



    
    
    # Instance:


    
    
    # Keypoint:







if __name__ == '__main__':
    main()
     