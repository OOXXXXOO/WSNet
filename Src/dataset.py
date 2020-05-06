# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    dataset.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/28 11:45:57 by winshare          #+#    #+#              #
#    Updated: 2020/04/30 13:42:01 by winshare         ###   ########.fr        #
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

import numpy as np



class DATASET(NETWORK,COCO,Dataset):
    def __init__(self):
        NETWORK.__init__(self)

        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #


        # --------------------------------- SFT Init --------------------------------- #
        if self.SFT_Enable:
            self.transforms=SFT(Mode=self.MissionType)
        else:
            self.transforms=Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                T.ToTensor()
            ])
        # ---------------------------------------------------------------------------- #
        #                         Smart Transform build Module                         #
        # ---------------------------------------------------------------------------- #

        # ----------------------------- COCO Dataset Init ---------------------------- #
        self.dataset_function=self.datasets_function_dict[self.MissionType][self.DataSetType]
        # --------------------------- DatasetFunction Index -------------------------- #
        if self.DataSetType == "CocoDetection":
            self.trainset=self.dataset_function(
                os.path.join(self.DataSet_Root,'/train2014'),
                self.Dataset_Train_file,
                transforms=self.transforms,
                Mode=self.MissionType
            )
            print("\ntrain dataset process done !\n")
            self.valset=self.dataset_function(
                os.path.join(self.DataSet_Root,'/val2014'),
                self.Dataset_Val_file,
                transforms=self.transforms,
                Mode=self.MissionType
            )
            print("\nval dataset process done !\n")
            print('\n\n-------------------------- COCO Dataset Init Done --------------------------\n\n')
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
            print('\n\n-------------------------- Costum_NPY_DataSet Dataset Init Done --------------------------\n\n')






        if self.DistributedDataParallel:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.trainset)
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.valset)
            print("-----DistributedDataParallel Sampler build done")
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=self.gpu_id)
            self.model_without_ddp = self.model.module
            

        if not self.DistributedDataParallel:

            self.train_sampler = torch.utils.data.RandomSampler(self.trainset)
            self.test_sampler = torch.utils.data.SequentialSampler(self.valset)
            print("-----DataSampler build done")


        self.train_batch_sampler = torch.utils.data.BatchSampler(
                self.train_sampler,
                self.BatchSize,
                drop_last=True
                )

        # ---------------------------------- loader ---------------------------------- #


        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, 
            batch_sampler=self.train_batch_sampler,
            num_workers=self.worker_num)

        print("---------------------- Training DataLoader Init Finish ---------------------")

        self.valloader = torch.utils.data.DataLoader(
            self.valset, 
            batch_size=1,
            sampler=self.test_sampler,
            num_workers=self.worker_num)
        
        # BUG INFO :
        """
        For Deeplab V3:
        Error: ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 256, 1, 1])
        
        The BatchSize Must >=2, BN Module will compute ave of Batch ,if Batch just have one data
        BN Module will be broken

        ValueError: batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
         sampler option is mutually exclusive with XXXXXXXXX

        """



        print("---------------------- Validation DataLoader Init Finish ---------------------")

        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #
        print("\n\n-------------------- DATASET Class Init Successful --------------------\n\n")


        # ---------------------------------------------------------------------------- #
        #                               DATASET Function                               #
        # ---------------------------------------------------------------------------- #

    






















    
