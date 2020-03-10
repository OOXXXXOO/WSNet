# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    transform.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/28 11:46:45 by winshare          #+#    #+#              #
#    Updated: 2020/03/10 19:14:11 by winshare         ###   ########.fr        #
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
import Src.Utils.Transform.custom as T
import os
import glob
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
from Src.Utils.Transform.custom import *
import torch
import torchvision.transforms.functional as F
# Transform_Function_Dict={
#                 "adjust_brightness":T.functional.adjust_brightness,
#                 "adjust_contrast":T.functional.adjust_contrast,
#                 "adjust_gamma":T.functional.adjust_gamma,
#                 "adjust_hue":T.functional.adjust_hue,
#                 "adjust_saturation":T.functional.adjust_saturation,
#                 "affine":T.functional.affine,
#                 "crop":T.functional.crop,
#                 "erase":T.functional.erase,
#                 "five_crop":T.functional.five_crop,
#                 "hflip":T.functional.hflip,
#                 "normalize":T.functional.normalize,
#                 "pad":T.functional.pad,
#                 "perspective":T.functional.perspective,
#                 "resize":T.functional.resize,
#                 "resized_crop":T.functional.resized_crop,
#                 "rotate":T.functional.rotate,
#                 "ten_crop":T.functional.ten_crop,
#                 "to_grayscale":T.functional.to_grayscale,
#                 "to_pil_image":T.functional.to_pil_image,
#                 "to_tensor":T.functional.to_tensor,
#                 "vflip":T.functional.vflip
#                 }


Transform_Class_Dict={
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


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, data):
        for t in self.transforms:
            # print(t)
            data = t(data)
        return data



class target_transform():
    def __init__(self,transformlist=None,randomdict=None):
        """
        target transform will recieve the transformlist & random transform parameter in image transform

        if the target is dictionry:
            all the position relate transform will be process

        if the target is ndarray:
            will process all transform after remove the pixel value relate transform

        """
        print("-----target transform")
        assert not transformlist==None,"Invalid Target Transform List"
        self.transformlist=transformlist
        self.TargetNeedPara={
                "Pad":T.Pad,#填充
                "Resize":T.Resize,
                "Scale":T.Scale,
                "TenCrop":T.TenCrop
        }
        self.random_transform={
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
        self.ndarray_transform={
                "Pad":T.Pad,
                "RandomAffine":T.RandomAffine,
                "RandomApply":T.RandomApply,
                "RandomChoice":T.RandomChoice,
                "RandomCrop":T.RandomCrop,
                "RandomErasing":T.RandomErasing,
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
                "ToTensor":T.ToTensor
        }
        self.dict_transform={
                "Pad":T.Pad,
                "RandomCrop":T.RandomCrop,
                "RandomErasing":T.RandomErasing,
                "RandomHorizontalFlip":T.RandomHorizontalFlip,
                "RandomPerspective":T.RandomPerspective,
                "RandomResizedCrop":T.RandomResizedCrop,
                "RandomRotation":T.RandomRotation,
                "RandomSizedCrop":T.RandomSizedCrop,
                "RandomVerticalFlip":T.RandomVerticalFlip,
                "Resize":T.Resize,
                "Scale":T.Scale,
                "TenCrop":T.TenCrop
        }
        self.target_transforms=[]


        # self.dict,self.arr=self.filter()
        """
        Temp change
        """

        self.target_dict_transforms=Compose(self.dict)
        self.target_arr_transforms=Compose(self.arr)







    def __call__(self,target):
        if isinstance(target,dict):
            return self.target_dict_transforms(target)
        else:
            return self.target_arr_transforms(target)
    
    def filter(self):

        """
        1. classify the dict/arr create dict & arr list
        2. classify the random or not ,init the para
        3. return the dict compose and arr compose
         
    
        """

        functionlist=[list(i.keys())[0] for i in self.transformlist]
        paralist=[list(i.values())[0] for i in self.transformlist]
        dict_=[]
        arr_=[]


        return dict_,arr_


class image_transform():
    def __init__(self,transformlist=None):
        assert not transformlist==None,"Invalid Target Transform List"
        print("-----image transform")
        self.image_transforms=[]
        functionlist=[list(i.keys())[0] for i in transformlist]
        paralist=[list(i.values())[0] for i in transformlist]
        for i in range(len(functionlist)):
            # print(paralist[i])
            if paralist[i]!='None':
                if type(paralist[i])==list:
                    self.image_transforms.append(Transform_Class_Dict[functionlist[i]](*paralist[i]))
                else:
                    self.image_transforms.append(Transform_Class_Dict[functionlist[i]](paralist[i]))

            else:
                self.image_transforms.append(Transform_Class_Dict[functionlist[i]]())
        self.image_transforms=Compose(self.image_transforms)
    
    def __call__(self,images):
        return self.image_transforms(images)


class GeneralTransform():
    def __init__(self,transformlist=None,Mission=None):
        """
        The official dataset getitem function dont have anyway to make transform support
        for all of them .
        So the General Transform class is a resolution :
        it can generate three way of transforms with one config file

        * transforms(image,target)
        * targets_transforms(target)
        * images_transforms(images)


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
            
            Mission: String of Mission Type like:
                "Detection" 
        
        Member:
        
        self.target_transform : list of target transform
        
        self.image_transform: list of image transform

        __call__:(image,target)
        return transformed image,target



        Random transform will be different  
        
        Whole Structurelike:
                                  |->ndarray_transform<---------|______
                       |->target->|->dict_vector_transform <----|filter|
        transformlist->|                                        |
                       |->image-->|->image_transform->|->random_para_dict

        The target in different mission just include two support way:
        image->image mask label | ndarray
        image->dict label       | dict
        
 
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
        assert not Mission in SupportMission,"Invalid Transform Dictionary"
        self.Mission=Mission
        self.image_transform=image_transform(transformlist=transformlist)
        #self.target_transform=target_transform(transformlist=transformlist,Mission=self.Mission)


    def __call__(self,images,targets):
        images=self.image_transform(images)
        # targets=self.target_transform(targets)
        """
        temprary change
        """
        targets=F.to_tensor(targets*255)
    
        return images,targets













def main():

    # ---------------------- Test Part for GeneralTransform ---------------------- #

    print(" ---------------------- Test Part for GeneralTransform ----------------------")
    
    # ---------------------------------- Usage: ---------------------------------- #
    



    DemoTransformDict=[
            {"ToTensor":"None"},
            {"Normalize":[[0.485,0.456,0.406],[0.229, 0.224, 0.225]]},
            # {"RandomResizedCrop":512},
            # {"RandomRotation":90},
        
        ]
















    Imagelist=glob.glob("/workspace/WSNets/Data/labelme/demo/train2014/*.jpg")

    images=[]
    for i in Imagelist:
        images.append(Image.open(i))
        


    # Detection:

    A=GeneralTransform(DemoTransformDict,"Detection")

    image_transform=A.image_transform
    for i in images:
        img=image_transform(i)

        # plt.imshow(np.array(img.numpy())),plt.show()
    # Segmentation:



    
    
    # Instance:


    
    
    # Keypoint:







if __name__ == '__main__':
    main()
     