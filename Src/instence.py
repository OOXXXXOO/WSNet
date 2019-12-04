# -*- coding: utf-8 -*-
# @Author: Winshare
# @Date:   2019-12-02 17:08:40
# @Last Modified by:   Winshare
# @Last Modified time: 2019-12-04 18:38:03

# Copyright 2019 Winshare
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


'''
@Author: Winshare
@Date: 2019-12-02 16:16:48
@LastEditTime: 2019-12-02 16:58:09
@LastEditors: Please set LastEditors
@Description: Instance
@FilePath: /WSNet/Src/instence.py
'''
from config_generator import cfg
from network_generator import NetworkGenerator
from dataset_generator import DatasetGenerator
from torch.utils.data import DataLoader
import torch
import os
import sys
import time
from general_train import train_one_epoch,evaluate
from Utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups





root=os.path.abspath(__file__)
print('instence work on ',root)

class Instence(NetworkGenerator,DatasetGenerator):
    def __init__(self,
    instence_id=0,
    config_dir='./cfg',
    ):  

        # ---------------------------------------------------------------------------- #
        #                                workspace info                                #
        # ---------------------------------------------------------------------------- #

        self.root=root
        print('root in :\n',os.path.join(self.root,'..'))
        sys.path.append(os.path.join(sys.path[0],'../'))
        print('workspace in:\n')
        for i in sys.path:
            print(i)
            
        DatasetGenerator.__init__(self)
        super(Instence,self).__init__()
        print('\n\n-----Instence Class Init-----\n\n')

        # ---------------------------------------------------------------------------- #
        #                                  dataloader                                  #
        # ---------------------------------------------------------------------------- #

        # ------------------------------ dataset object ------------------------------ #

        trainset=None
        valset=None
        
        if self.DefaultDataset:
            trainset=DatasetGenerator().DefaultDatasetFunction('train')
            valset=DatasetGenerator().DefaultDatasetFunction('val')
 
        # ----------------------------- DataLoader object ---------------------------- #
        
        if self.DistributedDataParallel:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)

        if not self.DistributedDataParallel:
            train_sampler = torch.utils.data.RandomSampler(trainset)
            test_sampler = torch.utils.data.SequentialSampler(valset)

        # ---------------------------------- Sampler --------------------------------- #
        
        if self.aspect_ratio_factor >= 0:
            group_ids = create_aspect_ratio_groups(trainset, k=self.aspect_ratio_factor)
            train_batch_sampler = GroupedBatchSampler(train_sampler,
            group_ids,
            self.BatchSize
            )
        else:
            train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler,
            self.BatchSize,
            drop_last=True
            )
        
        
        
        self.trainloader = torch.utils.data.DataLoader(
            trainset, 
            batch_sampler=train_batch_sampler,
            num_workers=self.worker_num,
            collate_fn=self.collate_fn)

        self.valloader = torch.utils.data.DataLoader(
            valset, 
            batch_size=self.BatchSize,
            sampler=test_sampler,
            num_workers=self.worker_num,
            collate_fn=self.collate_fn)



        # ---------------------------------------------------------------------------- #
        #                               Instance Function                              #
        # ---------------------------------------------------------------------------- #

    def InstenceInfo(self):
        print('\n\n-----Start with Instence ID',self.InstanceID,'-----\n\n')
        self.Enviroment_Info()
        self.DatasetInfo()
        self.NetWorkInfo()


    def default_train(self):
        print('\n\n----- Start Training -----\n\n')
        start_time = time.time()
        for epoch in range(0,self.epochs):
            train_one_epoch(
                self.model,
                self.optimizer,
                self.trainloader,
                self.device,
                epoch,
                10
            )
            self.lr_scheduler.step()
            
        






    def default_val(self):
        print('\n\n----- Val Processing -----\n\n')

    
    def inference(self):
        print('\n\n----- Inference Processing -----\n\n')

    def Evaluation(self):
        print('\n\n----- Evaluation Processing -----\n\n')
    
    






def main():
    
    instence=Instence()
    instence.default_train()



if __name__ == '__main__':
    main()
    