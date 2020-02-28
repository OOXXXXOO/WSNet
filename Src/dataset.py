# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    dataset.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/28 11:45:57 by winshare          #+#    #+#              #
#    Updated: 2020/02/28 12:21:36 by winshare         ###   ########.fr        #
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
from Utils.Transform.transform import Compose
from Utils.Transform.transform import GeneralTransform

from network import NETWORK
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

# ------------------------------ Local Reference ----------------------------- #

from Utils.COCO.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
import Utils.COCO.transforms as T
from Utils.COCO.coco_utils import ConvertCocoPolysToMask,_coco_remove_images_without_annotations
from Utils.COCO.coco_utils import get_coco_api_from_dataset
from Utils.COCO.coco_eval import CocoEvaluator
import Utils.COCO.utils







class DATASET(NETWORK,COCO,Dataset):
    def __init__(self):
        NETWORK.__init__(self)

        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #

        # ----------------------------- COCO Dataset Init ---------------------------- #
        self.dataset_function=self.datasets_function_dict[self.MissionType][self.DataSetType]
        if self.DataSetType == "CocoDetection":
            self.trainset=self.dataset_function(
                self.DataSet_Root+'/train2014',
                self.Dataset_Train_file,
                transforms=self.transforms
            )
            print("\ntrain dataset process done !\n")
            self.valset=self.dataset_function(
                self.DataSet_Root+'/val2014',
                self.Dataset_Val_file,
                transforms=self.transforms
            )
            print("\nval dataset process done !\n")
            print('\n\n-------------------------- COCO Dataset Init Done --------------------------\n\n')

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



        # ---------------------------------- Sampler --------------------------------- #
        
        if self.aspect_ratio_factor >= 0:
            self.group_ids = create_aspect_ratio_groups(self.trainset, k=self.aspect_ratio_factor)
            self.train_batch_sampler = GroupedBatchSampler(self.train_sampler,
            self.group_ids,
            self.BatchSize
            )
        else:
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

        print("---------------------- Training DataLoader Init Finish ---------------------")

        self.valloader = torch.utils.data.DataLoader(
            self.valset, 
            batch_size=self.BatchSize,
            sampler=self.test_sampler,
            num_workers=self.worker_num,
            collate_fn=self.collate_fn)

        print("---------------------- Validation DataLoader Init Finish ---------------------")


        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #
        print("\n\n----------------------- DATASET Class Init Successful ----------------------\n\n")


        # ---------------------------------------------------------------------------- #
        #                               DATASET Function                               #
        # ---------------------------------------------------------------------------- #




























    # def __getitem__(self,index):
    #     # assert self.DataSetProcessDone,"Invalid Dataset Object"
    #     # return self.getitem_map[self.DataSetType](index)
    #     if self.mode=='train':
    #         return self.trainset[index]
    #     if self.mode=='val':
    #         return self.valset[index]



    # def __len__(self):
    #     if self.mode=='train':
    #         return len(self.trainset)
    #     if self.mode=='val':
    #         return len(self.valset)
    
    
