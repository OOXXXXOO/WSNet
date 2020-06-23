# Copyright 2020 tanwenxuan
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
#    reference.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: tanwenxuan <tanwenxuan@student.42.fr>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/06/22 16:34:57 by tanwenxuan        #+#    #+#              #
#    Updated: 2020/06/22 16:34:57 by tanwenxuan       ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import torchvision.models as models
import torchvision.datasets as dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

# ------------------------- Official Pytorch library ------------------------- #

# future


# -------------------------- Official Keras library -------------------------- #


# future

# -------------------------- Official MXNet library -------------------------- #

# future


# -------------------------- Local Package Function -------------------------- #


from Data.COCO.coco import CocoDataSet


class reference():
    def __init__(self):
        """
        sample：
        前景色         背景色           颜色
        ---------------------------------------
        30                40              黑色
        31                41              红色
        32                42              绿色
        33                43              黃色
        34                44              蓝色
        35                45              洋红
        36                46              青色
        37                47              白色
        显示方式             　 意义
        ----------------------------------
        0                    终端默认设置
        1                    高亮显示
        22　　　　　　　　　　　非高亮显示
        4                    使用下划线
        24　　　　　　　　　　　去下划线
        5                    闪烁
        25　　　　　　　　　　　去闪烁
        7                    反白显示
        27　　　　　　　　　　　非反显
        8                    不可见
        28　　　　　　　　　　　可见
        
        例：
        \033[1;32;41m   #---1-高亮显示 31-前景色绿色  40-背景色红色---
        \033[0m         #---采用终端默认设置，即缺省颜色---      
        """
        print(
            "\033[46m# ----------------------------------------------------------------------------- #\033[0m\n"
            "\033[5;36m#        _             _            _             _          _          _       #\033[0m\n"        
            "\033[5;36m#       / /\      _   / /\         /\ \     _    /\ \       /\ \       / /\     #\033[0m\n"
            "\033[5;36m#      / / /    / /\ / /  \       /  \ \   /\_\ /  \ \      \_\ \     / /  \    #\033[0m\n"
            "\033[5;36m#     / / /    / / // / /\ \__   / /\ \ \_/ / // /\ \ \     /\__ \   / / /\ \__ #\033[0m\n"
            "\033[5;36m#    / / /_   / / // / /\ \___\ / / /\ \___/ // / /\ \_\   / /_ \ \ / / /\ \___\#\033[0m\n"
            "\033[5;36m#   / /_//_/\/ / / \ \ \ \/___// / /  \/____// /_/_ \/_/  / / /\ \ \  \ \ \/___/#\033[0m\n"
            "\033[5;36m#  / _______/\/ /   \ \ \     / / /    / / // /____/\    / / /  \/_/ \ \ \      #\033[0m\n"
            "\033[5;36m# / /  \____\  /_    \ \ \   / / /    / / // /\____\/   / / /    _    \ \ \     #\033[0m\n"
            "\033[5;36m#/_/ /\ \ /\ \//_/\__/ / /  / / /    / / // / /______  / / /    /_/\__/ / /     #\033[0m\n"
            "\033[5;36m#\_\//_/ /_/ / \ \/___/ /  / / /    / / // / /_______\/_/ /     \ \/___/ /      #\033[0m\n"
            "\033[5;36m#    \_\/\_\/   \_____\/   \/_/     \/_/ \/__________/\_\/       \_____\/       #\033[0m\n"
            "\033[46m# ----------------------------------------------------------------------------- #\033[0m\n"
        )

        print("# ---------------------------------------------------------------------------- #")
        print("#                          reference library init start                        #")
        print("# ---------------------------------------------------------------------------- #")

       
       
        # ============================================================================== Pytorch API Reference 
       
        # ---------------------------------------------------------------------------- #
        #                               DataSet Function                               #
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
                "CocoDetection":CocoDataSet,
                "VOC_Detection":dataset.VOCDetection
            },
            "Segmentation":{
                "VOC_Segmentation":dataset.VOCSegmentation,
                "Cityscapes":dataset.Cityscapes,
          
                "CocoSegmentation":CocoDataSet
            },
            "Caption":{
                "CocoCaptions":dataset.CocoCaptions
            },
            "InstenceSegmentation":{
                "CocoDetection":CocoDataSet
            }
        }
        self.dataset_support_list=self.datasets_function_dict.keys()



        # ---------------------------------------------------------------------------- #
        #                              Optimizer Function                              #
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
        #                                 Loss Function                                #
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
        #                            Learning Rate Scheduler                           #
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

        
        # ---------------------------------------------------------------------------- #
        #                               Network reference                              #
        # ---------------------------------------------------------------------------- #

        # --------------------------------- BackBone --------------------------------- #

        self.BackBoneDict={




        }

        # --------------------------- InstanceSegmentation --------------------------- #

        self.InstanceSegmentationDict={



        }

        # --------------------------------- Detection -------------------------------- #

        self.DetectionDict={




        }

        # ------------------------------- Segmentation ------------------------------- #

        self.SegmentationDict={



        }

        # ------------------------------------ MOT ----------------------------------- #
        self.MotDict={



        }

        # --------------------------------- Keypoint --------------------------------- #
        self.KeyPointDict={




        }

        # ------------------------------ Default Network ----------------------------- #
        self.DefaultNetwork={

            
        }

        # ============================================================================== Pytorch API Reference 
       











        print("# ---------------------------------------------------------------------------- #")
        print("#                              reference init done                             #")
        print("# ---------------------------------------------------------------------------- #")









def main():
    print("Unit Test for reference")







if __name__ == '__main__':
    main()
    