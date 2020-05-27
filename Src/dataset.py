# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    dataset.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/28 11:45:57 by winshare          #+#    #+#              #
#    Updated: 2020/05/27 19:59:46 by winshare         ###   ########.fr        #
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



# ---------------------------- Official Reference ---------------------------- #

import sys 
import os
print("---dataset.py workspace in :\n",sys.path)
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

# ------------------------------ Local Reference ----------------------------- #


from network import NETWORK
from Utils.Transform.STF import Compose
from Utils.Transform.STF import STF
import PIL.Image as Image
import numpy as np



class DATASET(NETWORK,COCO,Dataset):
    def __init__(self):
        NETWORK.__init__(self)
        self.basetransforms=Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #


        # --------------------------------- SFT Init --------------------------------- #
        if self.SFT_Enable:
            self.transforms=STF(mode=self.MissionType)
            print("# ----------------------------- SFT Module Enable ---------------------------- #")

        else:
            self.transforms=self.basetransforms
            print("# ---------------------------- SFT Module Disable ---------------------------- #")

        # ---------------------------------------------------------------------------- #
        #                         Smart Transform build Module                         #
        # ---------------------------------------------------------------------------- #

        # ----------------------------- COCO Dataset Init ---------------------------- #
        self.dataset_function=self.datasets_function_dict[self.MissionType][self.DataSetType]
        # --------------------------- DatasetFunction Index -------------------------- #
        if self.DataSetType == "CocoDetection":
            self.trainset=self.dataset_function(
                self.DataSet_Root,
                self.Dataset_Train_file,
                transforms=self.transforms,
                Mode=self.MissionType,
                train=True
            )

        
            print("\n# ------------------------ train dataset process done ------------------------ #\n")
            
            self.valset=self.dataset_function(
                self.DataSet_Root,
                self.Dataset_Val_file,
                transforms=self.transforms,
                Mode=self.MissionType,
                train=False
            )


            print("\n# ------------------------- val dataset process done ------------------------- #\n")

        # ---------------------------------- Sampler --------------------------------- #

        # ----------------------------- COCO Dataset Init ---------------------------- #
        
        # ------------------------------- Support list ------------------------------- #

        if self.DataSetType == "CocoCaptions":
            print("Not Support Now")
        if self.DataSetType == "ImageNet":
            print("Not Support Now")
        if self.DataSetType == "VOC_Detection":
            print("Not Support Now")
        if self.DataSetType == "Cityscapes":
            print("Not Support Now")
        if self.DataSetType == "VOC_Segmentation":
            print("Not Support Now")
        if self.DataSetType == "CIFAR100":
            print("Not Support Now")
        if self.DataSetType == "CIFAR10":
            print("Not Support Now")
        if self.DataSetType == "ImageNet":
            print("Not Support Now")
        if self.DataSetType == "Costum_NPY_DataSet":
            print('\n\n-----Start Costum_NPY_DataSet Buidling...')

            import glob
            npys=glob.glob("/workspace/data/38cloud-cloud-segmentation-in-satellite-images/ProcessedDataset/*.npy")
            npy=[]
            for n in npys:
                print('-----Read ',n)
                npy.extend(np.load(n,allow_pickle=True))
                print("Traing size:",len(npy))



            self.trainset=self.dataset_function(
                npy=self.NPY_Data,
                data_ratio=0.8,
                transforms=self.transforms
            )
            print("\ntrain dataset process done !\n")
            self.valset=self.dataset_function(
                npy=self.NPY_Data,
                forward=False,
                transforms=self.transforms,
                data_ratio=0.8
            )
            
            
            print("\nval dataset process done !\n")

            print("# ------------------- Costum_NPY_DataSet Dataset Init Done ------------------- #")






        if self.DistributedDataParallel:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.trainset)
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.valset)
            print("-----DistributedDataParallel Sampler build done")
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=self.gpu_id)
            self.model_without_ddp = self.model.module
            

        if not self.DistributedDataParallel:

            self.train_sampler = torch.utils.data.RandomSampler(self.trainset)
            self.test_sampler = torch.utils.data.SequentialSampler(self.valset)



        self.train_batch_sampler = torch.utils.data.BatchSampler(
                self.train_sampler,
                self.BatchSize,
                drop_last=True
                )

        # ---------------------------------- loader ---------------------------------- #


        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, 
            batch_sampler=self.train_batch_sampler,
            num_workers=self.worker_num,
            collate_fn=self.collate_fn)

        print("# ---------------------- Training DataLoader Init Finish --------------------- #")

        self.valloader = torch.utils.data.DataLoader(
            self.valset, 
            batch_size=1,
            sampler=self.test_sampler,
            num_workers=self.worker_num,
            collate_fn=self.collate_fn)
        
        # BUG INFO :
        """
        For Deeplab V3:
        Error: ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])
        
        The BatchSize Must >=2, BN Module will compute ave of Batch ,if Batch just have one data
        BN Module will be broken

        ValueError: batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
         sampler option is mutually exclusive with XXXXXXXXX

        """


        print("# --------------------- Validation DataLoader Init Finish -------------------- #")

        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #

        print("# ---------------------------------------------------------------------------- #")
        print("#                         DATASET Class Init Successful                        #")
        print("# ---------------------------------------------------------------------------- #")



        # ---------------------------------------------------------------------------- #
        #                               DATASET Function                               #
        # ---------------------------------------------------------------------------- #

    
    def paste(self,array,size):
        if isinstance(array,type(Image.Image)):
            result=Image.new("RGB",size)
            result.paste(array)
            return result
        if isinstance(array,type(np.array)):
            c,h,w=array.shape
            result=np.zeros((c,size[0],size[1]),dtype=np.float)
            result[:,:h,:w]=array[:,:,:]
            import matplotlib.pyplot as plt
            plt.imshow(result[0,:,:]),plt.show()
            return result

            
###############################################################



    # @staticmethod
    def collate_fn(self,batch):
        return tuple(zip(*batch))

# update 2020-05-27:
# 主要改动为添加拼接方法:
# 在需求Mask的任务中 为了在拼接过程中 防止坐标变换产生的图像坐标系变换
# 直接新建统一大小的Image 然后paste
# 在这个过程中区分任务类型:
# 1,目标检测:target不需要变换

# 2,实例分割  按照要求产生了Multi Object Mask (BatchSize*C*W*H)
# 需要对每一层ndarray做扩充
# 3,语义分割   按照要求对单层target mask (BatchSize*1,W,H)做 paste


        
#         images = [item[0] for item in batch]
#         targets = [item[1] for item in batch]
#         max_size=max([item.size for item in images])

#         img=[]
#         gt=[]
#         for image in images:
#             print(type(image))
#             print(image.size[-2:],"-----images-----",max_size[-2:])
#             if not image.size[-2:]==max_size[-2:]:
#                 img.append(self.paste(image,max_size[-2:]))
#         img=[self.basetransforms(i) for i in img]
#         if self.MissionType=="Segmentation":
#             for target in targets:
#                 print(target.size[-2:],"----targets------",max_size[-2:])
#                 if not target.size[-2:]==max_size[-2:]:
#                     gt.append(self.basetransforms(self.paste(target,max_size[-2:])))
        
#         if self.MissionType=="InstenceSegmentation":
# ###########################################################################


#             for target in targets:
#                 print(target["masks"].shape[-2:],"----targets------",max_size[-2:])
#                 if not target["masks"].shape[-2:]==max_size[-2:]:
#                     masks=target["masks"]
#                     print('sad',type(masks))
#                     gt.append(self.basetransforms(self.paste(target["masks"],max_size[-2:])))





    
