# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    model.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/28 11:46:08 by winshare          #+#    #+#              #
#    Updated: 2020/05/27 15:25:28 by winshare         ###   ########.fr        #
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
sys.path.append(sys.path[0][:-3])
for path in sys.path:
    print("-----root :",path)
import os

# ------------------------------- sys reference ------------------------------ #

import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch
import numpy as np
from tqdm import tqdm

# ---------------------------- official reference ---------------------------- #


from torch.utils.tensorboard import SummaryWriter

# ------------------------------ local reference ----------------------------- #

from Src.Utils.Summary.Analysis import summary
from Src.Utils.Evaluator.metrics import Evaluator
from Data.DataSets.COCO.coco import CocoEvaluation
from dataset import DATASET



















class MODEL(DATASET):
    def __init__(self,cfg,preestimation=False):
        self.configfile=cfg
        DATASET.__init__(self)
        self.default_input_size=(3,512,512)
        self.pre_estimation=preestimation

        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #
        """
        Init Process work for different Mission like :
        
        """

        # -------------------------------- train init -------------------------------- #
        
        


        # -------------------------------- train init -------------------------------- #
        
        # --------------------------------- eval init -------------------------------- #

        # self.evaluator =CocoEvaluation()
        self.eva_std_list={
            "Coco":CocoEvaluation
            # "Cityscapes":
            # "PascalVOC":
            # "SpaceNet":
        }

        # --------------------------------- eval init -------------------------------- #

        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #

        print("# ---------------------------------------------------------------------------- #")
        print("#                          Model Class Init Successful                         #")
        print("# ---------------------------------------------------------------------------- #")



    def train(self):
        print("# ---------------------------------------------------------------------------- #")
        print("#                                TRAIN START                                   #")
        print("# ---------------------------------------------------------------------------- #")
        # input_test=torch.randn(3,512,512
        self.model.to(self.device)
                
        # ---------------------------------------------------------------------------- #
        #                                Pre_estimation                                #
        # ---------------------------------------------------------------------------- #
        if self.pre_estimation:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(self.gpu_id))
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            str_="\n\n# ===== Used GRAM:,{used} GB/ {total} GB ===== #\n# ===== Free GRAM:,{free} GB/ {total} GB ===== #\n\n".format(
                used=float(meminfo.used)/1024**3,
                total=float(meminfo.total)/1024**3,
                free=float(meminfo.free)/1024**3
            )
            print(str_)
            # for i in range(deviceCount):
            #     handle = nvmlDeviceGetHandleByIndex(i)
            #     print("#-----Device", i, ":", nvmlDeviceGetName(handle)
            
            print("# ===== Compute Network Summary of Input:",self.default_input_size)
            _,total_size=summary(self.model,(3,512,512),batch_size=self.BatchSize,device=self.device)
            
            assert total_size<float(meminfo.free)/1024**3,"Estimation Network Size :"+\
                str(total_size.item())+\
                    "(GB) bigger than free GRAM : "+\
                        str(float(meminfo.free)/1024**3)+"(GB)"
        # ---------------------------------------------------------------------------- #
        #                                Pre_estimation                                #
        # ---------------------------------------------------------------------------- #
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
            self.writer = SummaryWriter(self.logdir)
        
        for epoch in range(self.epochs):
            self.one_epoch(epoch)
            self.val(epoch)
            
            








    def one_epoch(self,index):
        # ------------------------------ Train one epoch ----------------------------- #
        print("# =============================  epoch {index} ========================== #".format(index=index))
        for image,target in tqdm(self.trainloader):
            pass
            # print(image)
            # print(target)






























        

    def inference(self,data):
        print("# ---------------------------------------------------------------------------- #")
        print("#                                   INFERENCE                                  #")
        print("# ---------------------------------------------------------------------------- #")




    def val(self,index):
        """
        Validation Flow
        """

        print("# ---------------------------------------------------------------------------- #")
        print("#                                  VALIDATION                                  #")
        print("# ---------------------------------------------------------------------------- #")





    def save(self,epoch):
        print("-----save model in :") 
        modelstate={
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'preference':1
        }
        modelname=self.MissionType+"_"+self.NetType+str(time.asctime(time.localtime(time.time())))+'.pth'
        torch.save(modelstate,os.path.join(self.checkpoint,modelname))



    
def parser():
    parsers=argparse.ArgumentParser()
    parsers.add_argument("--config",default="./Config/segmantation.json", help="dir of config file")
    args = parsers.parse_args()
    return args



def main():
    args=parser()
    configfile=args.config
    model=MODEL(cfg=configfile)
    model.train()




if __name__ == '__main__':
    main()
    