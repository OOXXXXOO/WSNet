# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    STF.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/04/01 15:58:14 by winshare          #+#    #+#              #
#    Updated: 2020/04/29 17:51:52 by winshare         ###   ########.fr        #
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

# ---------------------------------------------------------------------------- #
#                             Smart Transform Class                            #
# ---------------------------------------------------------------------------- #
import sys
import os
sys.path.append(sys.path[0][:-19])
print(sys.path)
import glob
import numpy as np
import Src.Utils.Transform.box.data_aug as BT
import Src.Utils.Transform.mask.segmentation_transforms as MT
import PIL.Image as Image
import random



RandomRotateDegree=90
BaseSize=512
CropSize=512
FixScaleCropSize=512
FixedResize=512


mask_transform={
"Normalize":MT.Normalize(),
"ToTensor":MT.ToTensor(),
"RandomHorizontalFlip":MT.RandomHorizontalFlip(),
"RandomRotate":MT.RandomRotate(RandomRotateDegree),
"RandomGaussianBlur":MT.RandomGaussianBlur(),
"RandomScaleCrop":MT.RandomScaleCrop(BaseSize,CropSize),
"FixScaleCrop":MT.FixScaleCrop(FixScaleCropSize),
"FixedResize":MT.FixedResize(FixedResize),
}

mask_transform_list=list(mask_transform.values())

boxes_transform={
"RandomHorizontalFlip":BT.RandomHorizontalFlip,
"HorizontalFlip":BT.HorizontalFlip,
"RandomScale":BT.RandomScale,
"Scale":BT.Scale,
"RandomTranslate":BT.RandomTranslate,
"Translate":BT.Translate,
"RandomRotate":BT.RandomRotate,
"Rotate":BT.Rotate,
"RandomShear":BT.RandomShear,
"Shear":BT.Shear,
"Resize":BT.Resize,
"RandomHSV":BT.RandomHSV 
}

boxes_transform_list=list(boxes_transform.values())

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self,data):
        for t in self.transforms:
            print("======>>>>",t)
            data = t(data)
        return data


class STF():
    def __init__(self,mode):
        """
        Smart Transform is 2nd-gen Auto-Transform Class
        """

        print("# ---------------------------------------------------------------------------- #")
        print("#                    Smart Automatic Transform Process Init                    #")
        print("# ---------------------------------------------------------------------------- #")


        self.SupportMission={
            "Detection":self.DetectionTransform,
            "Segmentation":self.SegmentationTransform,
            "InstanceSegmentation":self.InstanceSegmentationTransform,
            # "KeyPoint",# Not support now
            # "Caption"
        }
        self.mode=mode
        assert mode in self.SupportMission.keys(),"Invalid `Mode` in STF process : "+mode
    def __call__(self,image,target):
        """

        * Detection
            target dict:             
            - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W
            - labels (Int64Tensor[N]): the class label for each ground-truth box
        * InstanceSegmentation
            target dict :
            - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
            between 0 and H and 0 and W
            - labels (Int64Tensor[N]): the class label for each ground-truth box
            - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance
        * Segmentation
            target dict :
            Output [(Batch_Size),W,H,CLASS_NUM] argmax(Axis=1) with w*h*c => [(Batch_Size),W,H]
            Target [(Batch_Size),W,H] value is classes index
        """
        assert isinstance(image,(np.ndarray,Image.Image)),"Invalid Image Input type :"+str(type(image))
        assert isinstance(target,dict),"Invalid target format :"+str(type(target))+" Target must be dict"
        image,target=self.SupportMission[self.mode](image,target)
        return image,target

    # ---------------------------- Transform Function ---------------------------- #

    def InstanceSegmentationTransform(self,image,target):
        instancemasks=target["masks"]
        boxes=target["boxes"]
        labels=target["labels"]



        return image,target

    def DetectionTransform(self,image,target):
        boxes=target["boxes"]
        labels=target["labels"]
        
        
        return image,target

    def SegmentationTransform(self,image,target):
        

        segmasks=target["masks"]
        sample={}
        sample["image"]=image
        sample["label"]=segmasks
        
        INDEX=random.randint(2,len(mask_transform_list)-1)
        Transformlist=[]
        Transformlist.append(mask_transform_list[INDEX])
        Transformlist.extend(mask_transform_list[:2])
        transform=Compose(Transformlist)
        result=transform(sample)
        image=result["image"]
        target=result["label"]
        return image,target


def main():
    Transform=STF("Segmentation")
    
    anno=np.ones((100,100,3),dtype=np.uint8)
    img=anno.copy()*255
    anno[20:70,20:70,:]=127
    boxes=[[20,20,70,70]]
    # print(anno.shape)
    # anno=Image.fromarray(anno)
    # img=Image.fromarray(img)
    # anno.show()
    target={}
    target["masks"]=anno
    image,target=Transform(img,target)
    # print(image,target)
    import matplotlib.pyplot as plt
    print(image.size(),target.size())
    
    plt.imshow(np.uint8(target.numpy())),plt.show()



if __name__ == '__main__':
    main()
    