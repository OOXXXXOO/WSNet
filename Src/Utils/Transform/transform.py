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

# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    transform.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <winshare@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/18 15:25:09 by winshare          #+#    #+#              #
#    Updated: 2020/02/18 15:25:09 by winshare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import sys
import torchvision.transforms as T
import os
import glob
import PIL.Image as Image


Transform_Function_Dict={
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
            data = t(data)
        return data



class target_transform():
    def __init__(self,transformlist=None):
        print("-----target transform")
        assert not transformlist==None,"Invalid Target Transform List"
        self.TargetNeedPara={
            "Pad":T.Pad,
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
        self.target_transforms=[]
        functionlist=[list(i.keys())[0] for i in transformlist]
        paralist=[list(i.values())[0] for i in transformlist]
        for i in range(len(functionlist)):
            if functionlist[i] in self.random_transform.keys():
                if type(paralist[i])==list:
                    
                    function=Transform_Class_Dict[functionlist[i]](*paralist[i])
                    
                else:
                    para=paralist[i]
                    function=Transform_Class_Dict[functionlist[i]](para)

                
                A=function.__repr__()
                print("-----Random Function & Para:",A)

            if functionlist[i] in self.TargetNeedPara.keys():
                para=paralist[i]
                print(para)
            
            else:
                print("-----Warning : Unnecessary Transform for Target has been deprecated",functionlist[i])
        self.target_transforms=Compose(self.target_transforms)
    def __call__(self,target):
        return self.target_transforms(target)



class image_transform():
    def __init__(self,transformlist=None):
        assert not transformlist==None,"Invalid Target Transform List"
        print("-----image transform")
        self.image_transforms=[]
        functionlist=[list(i.keys())[0] for i in transformlist]
        paralist=[list(i.values())[0] for i in transformlist]
        for i in range(len(functionlist)):
            print(paralist[i])
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
        self.image_transform=image_transform(transformlist=transformlist)
        self.target_transform=target_transform(transformlist=transformlist)


    def __call__(self,images,targets):
        images=self.image_transform(images)
        targets=self.target_transform(targets)
        return images,targets













def main():

    # ---------------------- Test Part for GeneralTransform ---------------------- #

    print(" ---------------------- Test Part for GeneralTransform ----------------------")
    
    # ---------------------------------- Usage: ---------------------------------- #
    
    DemoTransformDict=[
            {"Normalize":[[0.485,0.456,0.406],[0.229, 0.224, 0.225]]},
            {"RandomResizedCrop":512},
            {"RandomRotation":90},
            {"ToTensor":"None"}
        ]
    Imagelist=glob.glob("/workspace/WSNets/Data/labelme/demo/train2014/*.jpg")

    images=[]
    for i in Imagelist:
        images.append(Image.open(i))


    # Detection:

    A=GeneralTransform(DemoTransformDict,"Detection")

    
    # Segmentation:



    
    
    # Instance:


    
    
    # Keypoint:







if __name__ == '__main__':
    main()
     