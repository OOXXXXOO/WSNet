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
#    dataset.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <winshare@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/18 15:23:49 by winshare          #+#    #+#              #
#    Updated: 2020/02/18 15:23:49 by winshare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #



import sys 
print("---dataset.py workspace in :\n",sys.path)
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch
from Utils.Transform.transform import Compose
from network import NETWORK
from pycocotools.coco import COCO

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
            print('\n\n-------------------------- COCO Dataset Init Done --------------------------')

        # ----------------------------- COCO Dataset Init ---------------------------- #










        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #
        print("\n\n----------------------- DATASET Class Init Successful ----------------------\n\n")

    def __getitem__(self,index):
        # assert self.DataSetProcessDone,"Invalid Dataset Object"
        # return self.getitem_map[self.DataSetType](index)
        if self.mode=='train':
            return self.trainset[index]
        if self.mode=='val':
            return self.valset[index]



    def __len__(self):
        if self.mode=='train':
            return len(self.trainset)
        if self.mode=='val':
            return len(self.valset)
    
    
