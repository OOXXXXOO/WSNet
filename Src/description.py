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
#    description.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <winshare@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/06/08 15:42:45 by winshare          #+#    #+#              #
#    Updated: 2020/06/08 15:42:45 by winshare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #



from Data.DataSets.NPY.segmentation_dataset import Costum_NPY_DataSet
from Data.DataSets.CitysCapes.cityscapes import CityscapesSegmentation
from Data.DataSets.COCO.coco import CocoDataset
from Data.DataSets.PascalVoc.pascal import VOCSegmentation
from Src.Nets.BackBone.efficientnet.model import EfficientNet
from Src.Nets.BackBone.xception import AlignedXception as xception
from Src.Nets.BackBone.mobilenetv3 import MobileNetV3_Large,MobileNetV3_Small


import torchvision.models as models
import torchvision.datasets as dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class Description():
    def __init__(self):
        
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
                "CocoDetection":CocoDataset,
                "VOC_Detection":dataset.VOCDetection
            },
            "Segmentation":{
                "VOC_Segmentation":dataset.VOCSegmentation,
                "Cityscapes":dataset.Cityscapes,
                "Costum_NPY_DataSet":Costum_NPY_DataSet,
                "CocoSegmentation":CocoDataset
            },
            "Caption":{
                "CocoCaptions":dataset.CocoCaptions
            },
            "InstenceSegmentation":{
                "CocoDetection":CocoDataset
            }
        }
        self.dataset_support_list=self.datasets_function_dict.keys()


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


        self.BackBoneDict={
            
            # ------------------------------ Official Model ------------------------------ #

            "resnet18" :models.resnet18,
            "alexnet" :models.alexnet,
            "vgg16" :models.vgg16,
            "squeezenet":models.squeezenet1_0,
            "densenet": models.densenet161,
            "inception":models.inception_v3,
            "googlenet ": models.googlenet,
            "shufflenet ":models.shufflenet_v2_x1_0,
            "mobilenet ": models.mobilenet_v2,
            "resnext50_32x4d":models.resnext50_32x4d,
            "wide_resnet50_2" :models.wide_resnet50_2,
            "mnasnet": models.mnasnet1_0,

            # ------------------------------- Custom Model ------------------------------- #
            'efficientnet-b0':EfficientNet.from_name('efficientnet-b0'),
            'efficientnet-b1':EfficientNet.from_name('efficientnet-b1'),
            'efficientnet-b2':EfficientNet.from_name('efficientnet-b2'),
            'efficientnet-b3':EfficientNet.from_name('efficientnet-b3'),
            'efficientnet-b4':EfficientNet.from_name('efficientnet-b4'),
            'efficientnet-b5':EfficientNet.from_name('efficientnet-b5'),
            'efficientnet-b6':EfficientNet.from_name('efficientnet-b6'),
            'efficientnet-b7':EfficientNet.from_name('efficientnet-b7'),
            "xception":xception,
            "mobilenetv3_s":MobileNetV3_Small,
            "mobilenetv3_l":MobileNetV3_Large
        }

        print("# ---------------------------------------------------------------------------- #")
        print("#                             Description Init Done                            #")
        print("# ---------------------------------------------------------------------------- #")