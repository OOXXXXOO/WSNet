# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    model.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/28 11:46:08 by winshare          #+#    #+#              #
#    Updated: 2020/05/20 19:25:53 by winshare         ###   ########.fr        #
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

from Src.Utils.Evaluator.metrics import Evaluator
from Data.DataSets.COCO.coco import CocoEvaluation
from dataset import DATASET
from torch.utils.tensorboard import SummaryWriter

# ------------------------------ local reference ----------------------------- #






















class MODEL(DATASET):
    def __init__(self,cfg):
        self.configfile=cfg
        DATASET.__init__(self)

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
        """
        Train Flow :

        1.Init DataSet(Transform)
        2.Init Optimizer


        """
        print("# ---------------------------------------------------------------------------- #")
        print("#                                TRAIN START                                   #")
        print("# ---------------------------------------------------------------------------- #")
        if os.path.exists(self.logdir):
            os.removedirs(self.logdir)
            os.makedirs(self.logdir)
        else:
            os.makedirs(self.logdir)
            self.writer = SummaryWriter(self.logdir)
            
            





































        

    def inference(self,data):
        print("# ---------------------------------------------------------------------------- #")
        print("#                                   INFERENCE                                  #")
        print("# ---------------------------------------------------------------------------- #")




    def val(self):
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
    parsers.add_argument("--config",default="./Config/Demo.json", help="dir of config file")
    args = parsers.parse_args()
    return args



def main():
    args=parser()
    configfile=args.config
    model=MODEL(cfg=configfile)
    model.train()




if __name__ == '__main__':
    main()
    