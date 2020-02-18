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
#    config.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <winshare@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/18 15:22:11 by winshare          #+#    #+#              #
#    Updated: 2020/02/18 15:22:25 by winshare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #





import sys
import os
import json
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as T
import torchvision.datasets as dataset

class CFG():
    def __init__(self):

        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #
        print("Start Read Config :",self.configfile)





        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #

        # ---------------------------------------------------------------------------- #
        #                          Pytorch Function Dictionary                         #
        # ---------------------------------------------------------------------------- #

        self.datasets_function_dict={
            "Classification":{
                "MINST":dataset.MNIST,
                "FashionMINST":dataset.FashionMNIST,
                "KMINST":dataset.KMNIST,
                "EMINST":dataset.EMNIST,
                "CIFAR10":dataset.CIFAR10,
                "CIFAR100":dataset.CIFAR100,
                "ImageNet":dataset.ImageNet
            },
            "Detection":{
                "CocoDetection":dataset.CocoDetection,
                "VOC_Detection":dataset.VOCDetection
            },
            "Segmentation":{
                "VOC_Segmentation":dataset.VOCSegmentation,
                "Cityscapes":dataset.Cityscapes
            },
            "Caption":{
                "CocoCaptions":dataset.CocoCaptions
            },
            "Instance":{
                "CocoInstance":dataset.CocoDetection
            }
        }
        self.dataset_support_list=self.datasets_function_dict.keys()
        print("\n-----Support Mission Mode: ",self.dataset_support_list)

        # ---------------------------------------------------------------------------- #
        self.OptimDict={
           "SGD":optim.SGD,                                                                                                                                              
           "ASGD":optim.ASGD,
           "Adam":optim.Adam,
           "Adadelta":optim.Adadelta,
           "Adagrad":optim.Adagrad,
           "AdamW":optim.AdamW,
           "LBFGS":optim.LBFGS,
           "RMSprop":optim.RMSprop,
           "SparseAdam":optim.SparseAdam,
           "Adamax":optim.Adamax
        }
        # ---------------------------------------------------------------------------- #
        self.Loss_Function_Dict={
            "AdaptiveLogSoftmaxWithLoss":nn.AdaptiveLogSoftmaxWithLoss
            ,"BCELoss":nn.BCELoss 
            ,"BCEWithLogitsLoss":nn.BCEWithLogitsLoss 
            ,"CosineEmbeddingLoss":nn.CosineEmbeddingLoss 
            ,"CrossEntropyLoss":nn.CrossEntropyLoss 
            ,"CTCLoss":nn.CTCLoss 
            ,"cosine_embedding_loss":F.cosine_embedding_loss 
            ,"ctc_loss":F.ctc_loss
            ,"hinge_embedding_loss":F.hinge_embedding_loss 
            ,"l1_loss":F.l1_loss 
            ,"margin_ranking_loss":F.margin_ranking_loss 
            ,"mse_loss":F.mse_loss 
            ,"multi_margin_loss":F.mse_loss 
            ,"multilabel_margin_loss":F.multilabel_margin_loss 
            ,"multilabel_soft_margin_loss":F.multilabel_margin_loss 
            ,"nll_loss":F.nll_loss 
            ,"poisson_nll_loss":F.poisson_nll_loss 
            ,"smooth_l1_loss":F.smooth_l1_loss 
            ,"soft_margin_loss":F.soft_margin_loss 
            ,"triplet_margin_loss":F.triplet_margin_loss 
            ,"HingeEmbeddingLoss":nn.HingeEmbeddingLoss 
            ,"KLDivLoss":nn.KLDivLoss 
            ,"L1Loss":nn.L1Loss 
            ,"MarginRankingLoss":nn.MarginRankingLoss 
            ,"MSELoss":nn.MSELoss 
            ,"MultiLabelMarginLoss":nn.MultiLabelMarginLoss 
            ,"MultiLabelSoftMarginLoss":nn.MultiLabelSoftMarginLoss 
            ,"MultiMarginLoss":nn.MultiMarginLoss 
            ,"NLLLoss":nn.MultiMarginLoss 
            ,"PoissonNLLLoss":nn.PoissonNLLLoss 
            ,"SmoothL1Loss":nn.SmoothL1Loss 
            ,"SoftMarginLoss":nn.SoftMarginLoss 
            ,"TripletMarginLoss":nn.TripletMarginLoss
        }
        # ---------------------------------------------------------------------------- #

        self.Transform_Dict={
            """
            General Transform add
            """
        }
        # ---------------------------------------------------------------------------- #

        self.Lr_Dict={
            "StepLR":optim.lr_scheduler.StepLR,
            "MultiStepLR":optim.lr_scheduler.MultiStepLR,
            "ExponentialLR":optim.lr_scheduler.ExponentialLR,
            "CosineAnnealingLR":optim.lr_scheduler.CosineAnnealingLR,
            "ReduceLROnPlateau":optim.lr_scheduler.ReduceLROnPlateau,
            "CyclicLR":optim.lr_scheduler.CyclicLR,
            "OneCycleLR":optim.lr_scheduler.OneCycleLR,
            "CosineAnnealingWarmRestarts":optim.lr_scheduler.CosineAnnealingWarmRestarts
        }

        self.__configfile=self.configfile
        self.__json=json.load(open(self.__configfile,'r'))
        self.usegpu=False


        self.MissionType=self.__json['MissionType']
        self.InstanceID=self.__json['instance_id']
        self.Content=self.__json['content']


        print("\n\n---------------------- Configure Class Init Successful ---------------------\n\n")


