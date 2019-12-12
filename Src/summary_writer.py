# -*- coding: utf-8 -*-
# @Author: Winshare
# @Date:   2019-12-03 16:44:55
# @Last Modified by:   Winshare
# @Last Modified time: 2019-12-10 16:25:29

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

class Nets_Writer():
    def __init__(self):
        super(Nets_Writer,self).__init__(logdir='./log',comments='experiment')
        self.writer=SummaryWriter(log_dir=logdir,comment=comments)
        self.paradict={
            "Detection":[
                mAP,
                F1,
                BoxLoss,
                Classification_Loss,
                Loss
                ]
            "Segmentation":[
                
            ]
            
            


            
        }

    