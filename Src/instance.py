# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    instance.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/28 11:46:08 by winshare          #+#    #+#              #
<<<<<<< HEAD
#    Updated: 2020/02/28 12:21:35 by winshare         ###   ########.fr        #
=======
#    Updated: 2020/03/04 22:09:13 by winshare         ###   ########.fr        #
>>>>>>> push test
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
<<<<<<< HEAD
import os
from dataset import DATASET
class INSTANCE(DATASET):
    def __init__(self,cfg):
        self.configfile=cfg
        DATASET.__init__(self,cfg=cfg)
=======
sys.path.append(sys.path[0][:-3])
import os
from dataset import DATASET
import argparse
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import torch
import numpy as np
from tqdm import tqdm

from Src.Utils.Evaluator.metrics import Evaluator

import time

class INSTANCE(DATASET):
    def __init__(self,cfg):
        self.configfile=cfg
        DATASET.__init__(self)
>>>>>>> push test
        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #


<<<<<<< HEAD







        # -------------------------------- DataLoader -------------------------------- #
        print("-------------------------------- DataLoader --------------------------------")





















=======
        self.evaluator = Evaluator(self.class_num)
>>>>>>> push test

        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #
        print("\n\n---------------------- INSTANCE Class Init Successful ----------------------\n\n")


<<<<<<< HEAD

def parser():
    parsers=argparse.ArgumentParser()
    parsers.add_argument("--config",default="Src/Config/Demo.json", help="dir of config file")
=======
        

    def train(self):
        torch.cuda.empty_cache()
        for i in range(5):
            print("#####------------------------------------------------------------------#####")

        print("------------- All Preprocess flow has been done Start Training ------------")

        for i in range(5):
            print("#####------------------------------------------------------------------#####")
        """
        resume
        part
        """
        
        
        lossavg=[]
        self.model.to(self.device)
        self.model.train()
        step=0
        self.ACC=0
        train_loss=0
        
        self.writer = SummaryWriter(log_dir=self.logdir,comment="Instance "+self.InstanceID+self.MissionType)
        # boardcommand="tensorboard --logdir="+self.logdir
        # os.system(boardcommand)

        for epoch in range(self.epochs):
            print('-----Epoch :',epoch)
            for images,targets in self.trainloader:
  
      
                # print("target:",targets.type(),targets.size())
                # ----------------------------------- Epoch ---------------------------------- #

                if self.usegpu:
                    images,targets=images.to(self.device),targets.to(self.device) 
                # self.lr_scheduler(self.optimizer,step)
                self.optimizer.zero_grad()

                if self.DefaultNetwork:
                    logits=self.model(images)['out']
                else:
                    logits=self.model(images)
                """
                Default Network Output the OrderedDict:
                {
                    'out':output_tensor
                }
                Custom Network will output tensor directly
                """
                logits=logits.argmax(1).unsqueeze(1).float().requires_grad_()
                
                # print("logits:",logits.type(),logits.size())
                # print("target:",targets.type(),targets.size())


                loss=self.Loss_Function(logits,targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                lossavg.append(loss.item())
                print('---loss:',loss.item(),'---step :',step)
                self.writer.add_scalar('train/total_loss_iter', loss.item(), i + len(self.trainloader) * epoch)
                step+=1
                # ----------------------------------- Epoch ---------------------------------- #
            
            
            self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)

            if step%100==0:
                self.visualize_image(images,targets,logits,step)

            ACC=self.val(epoch)    
            if self.ACC<ACC:
                self.save(epoch)
            else:
                self.ACC=ACC
                
            



    def visualize_image(self, image, target, output, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        self.writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(torch.max(output[:3], 1)[1].detach().cpu().numpy()
                                                    , 3, normalize=False, range=(0, 255))
        self.writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(torch.squeeze(target[:3], 1).detach().cpu().numpy()
                                                        , 3, normalize=False, range=(0, 255))
        self.writer.add_image('Groundtruth label', grid_image, global_step)

    def resume(self):
        print("-----load status")

    def val(self,epoch):
        print("-----Validation Process：")
        
        self.model.eval()
        self.evaluator.reset()
        test_loss = 0.0
        for images,targets in tqdm(self.valloader):
            if self.usegpu:
                images,targets=images.to(self.device),targets.to(self.device)
                with torch.no_grad():
                    if self.DefaultNetwork:
                        logits=self.model(images)['out']
                    else:
                        logits=self.model(images)
                logits=logits.argmax(1).unsqueeze(1).float()
                loss=self.Loss_Function(logits,targets)
                test_loss += loss.item()
                pred=logits.cpu().detach().numpy()
                # pred = logits.cpu().detach().numpy()
                target = targets.cpu().numpy()
                # pred = np.unsqeez np.argmax(pred,1)
                self.evaluator.add_batch(target.astype(np.int64), pred.astype(np.int64))

        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        self.mIoU=mIoU
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        print('Validation:')
        print('[Epoch: %d]' % (epoch))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Loss: %.3f' % test_loss)
        return Acc



    def save(self,epoch):
        print("-----save model in :") 
        modelstate={
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'ACC':self.ACC
        }
        modelname=self.MissionType+"_"+self.NetType+"_mIOU："+str(self.mIoU)+str(time.asctime(time.localtime(time.time())))+'.pth'
        torch.save(modelstate,os.path.join(self.checkpoint,modelname))



    def inference(self):
        print("-----inference process :")

    
def parser():
    parsers=argparse.ArgumentParser()
    parsers.add_argument("--config",default="./Config/Demo.json", help="dir of config file")
>>>>>>> push test
    args = parsers.parse_args()
    return args



def main():
    args=parser()
    configfile=args.config
    print(configfile)
<<<<<<< HEAD
    instence=Instence(configfile=configfile)
    instence.default_train()
=======
    instence=INSTANCE(cfg=configfile)
    instence.train()

>>>>>>> push test



if __name__ == '__main__':
    main()
    