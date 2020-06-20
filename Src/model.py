# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    model.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <winshare@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/28 11:46:08 by winshare          #+#    #+#              #
#    Updated: 2020/06/16 18:45:41 by winshare         ###   ########.fr        #
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

import torch
import numpy as np
from tqdm import tqdm
import math

# ---------------------------- official reference ---------------------------- #



# ------------------------------ local reference ----------------------------- #

from Src.Utils.Summary.Analysis import summary
from Src.Utils.Evaluator.metrics import Evaluator
from Data.DataSets.COCO.coco import CocoEvaluation
from dataset import DATASET
from trainer import TRAINER
from evaluator import EVALATOR



class MODEL(DATASET):
    def __init__(self,cfg):
        self.configfile=cfg
        DATASET.__init__(self)
        self.default_input_size=(3,512,512)
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
        if self.pre_estimation:
            self.model.eval()
            self.model.cuda()
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
            
            print("# ===== Compute Network Summary of Input:",self.default_input_size)
            _,total_size=summary(self.model,(3,512,512),batch_size=self.BatchSize,device=self.device)
            
            assert total_size<float(meminfo.free)/1024**3,"Estimation Network Size :"+\
                str(total_size.item())+\
                    "(GB) bigger than free GRAM : "+\
                        str(float(meminfo.free)/1024**3)+"(GB)"
        
        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #

        print("# ---------------------------------------------------------------------------- #")
        print("#                          Model Class Init Successful                         #")
        print("# ---------------------------------------------------------------------------- #")





    def copy_to_gpu(self,images,targets):
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return images,targets


    def inference(self,data):
        print("# ---------------------------------------------------------------------------- #")
        print("#                                   INFERENCE                                  #")
        print("# ---------------------------------------------------------------------------- #")






    def save(self,epoch):

        print("# =====Save model in :") 
        modelstate={
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'performance':1
        }
        modelname=self.MissionType+"_"+self.NetType+str(time.asctime(time.localtime(time.time())))+'.pth'
        torch.save(modelstate,os.path.join(self.checkpoint,modelname))


































    def train(self):
        print("# ---------------------------------------------------------------------------- #")
        print("#                                TRAIN START                                   #")
        print("# ---------------------------------------------------------------------------- #")
        self.model.cuda()
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        self.writer = SummaryWriter(self.logdir)
        self.global_step=0
        for epoch in range(self.epochs):
            self.one_epoch(epoch)
            self.val(epoch)
            
    

    def val(self,index):
        """
        Validation Flow
        """
        print("# ---------------------------------------------------------------------------- #")
        print("#                                  VALIDATION                                  #")
        print("# ---------------------------------------------------------------------------- #")
        print("# ============================= validation epoch {index} ========================== #".format(index=index))
        self.model.val()
        bar=tqdm(self.valloader,dynamic_ncols=True)
        if not self.MissionType=="Segmentation" or not self.DefaultNetwork:
            with torch.no_grad():
                 for image,target in bar:
                    if self.devices=="GPU":
                        image,target=self.copy_to_gpu(image,target)
                    output=self.model(image)
                    # ----------------- Compute the Model Accuracy & Performance ----------------- #
















    def one_epoch(self,index):
        # ------------------------------ Train one epoch ----------------------------- #
        print("# ============================= train epoch {index} ========================== #".format(index=index))
        # --------------------------- General Epoch Module --------------------------- #
        self.model.train()
        bar=tqdm(self.trainloader,dynamic_ncols=True)
        for image,target in bar:
            if self.devices=="GPU":
                image,target=self.copy_to_gpu(image,target)    
            # ------------------------------- Loss function ------------------------------ #     
            if not self.MissionType=="Segmentation" or not self.DefaultNetwork:
                lossdict=self.model(image,target)
                losses = sum(loss for loss in lossdict.values())
                lossstr={k:v.item() for k,v in lossdict.items()}
            else:
                output=self.model(image)
                loss=self.Loss_Function(output,target)
                lossstr=loss.item()
        
            self.writer.add_scalars(self.NetType+'_Loss Function',lossstr,global_step=self.global_step)
        
            # ------------------------------ Inference Once ------------------------------ #
            
            
            # -------------------------------- Output Loss ------------------------------- #

            information="# Step : {step} |loss : {loss} |\n".format(step=self.global_step,loss=str(lossstr))
            bar.set_description(information)

            # --------------------------------- Backward --------------------------------- #

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            
            # -------------------------- Add log to Tensorboard -------------------------- #

            self.global_step+=1
                



























    
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
    