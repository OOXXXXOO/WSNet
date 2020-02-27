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
from Utils.Transform.transform import GeneralTransform
class CFG():
    def __init__(self):

        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #
        for i in range(5):
            print("#####------------------------------------------------------------------#####")
        print("#####------------------------- Framework Start ------------------------#####")
        for i in range(5):
            print("#####------------------------------------------------------------------#####")

       
        print("-----Read Config :\n\n",self.configfile)





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
                "Cityscapes":dataset.Cityscapes,
                "CocoDetection":dataset.CocoDetection
            },
            "Caption":{
                "CocoCaptions":dataset.CocoCaptions
            },
            "Instance":{
                "CocoDetection":dataset.CocoDetection
            }
        }
        self.dataset_support_list=self.datasets_function_dict.keys()
        print("\n-----Support Mission Mode:\n\n",self.dataset_support_list)

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
        #                               Config in 3 Level                              #
        # ---------------------------------------------------------------------------- #


        # -------------------------------- File Level -------------------------------- #
        self.__configfile=self.configfile
        self.__json=json.load(open(self.__configfile,'r'))
        self.usegpu=False

        self.MissionType=self.__json['MissionType']
        self.InstanceID=self.__json['instance_id']
        self.Content=self.__json['content']


        # ------------------------------- Second Level ------------------------------- #
        self.Net=self.Content['Net']
        self.DataSetConfig=self.Content['Dataset']
        self.Config=self.Content['Config']

        print('\n\n---------------------------------- config ----------------------------------')
        print('----- Network Config : ')
        self.print_dict(self.Net)
        print('\n\n----- Dataset Config : ')
        self.print_dict(self.DataSetConfig)
        print('\n\n----- General Config : ')
        self.print_dict(self.Config)
        print('---------------------------------- config ----------------------------------')


        # -------------------------------- Third Level ------------------------------- #
        # ---------------------------------------------------------------------------- #
        #                                      NET                                     #
        # ---------------------------------------------------------------------------- #

        # self.NetType=self.Net['NetType']
        self.DefaultNetwork=self.Net["DefaultNetwork"]
        
    
        self.BatchSize=self.Net['BatchSize']
        if self.Net['BackBone']=='None':
            self.BackBone=None
        else:
            self.BackBone=self.Net['BackBone']
        

        # --------------------------------- Optimizer -------------------------------- #

        self.optimizer=self.OptimDict[self.Net['Optimizer']]
        self.learning_rate=self.Net['learning_rate']
        self.momentum=self.Net['momentum']
        self.weight_decay=self.Net['weight_decay']

        # ------------------------------- lr_scheduler ------------------------------- #

        self.lr_scheduler=self.Net['lr_scheduler']
        self.lr_steps=self.Net['lr_steps']
        self.lr_gamma=self.Net['lr_gamma']
        self.lr_scheduler=self.Lr_Dict[self.lr_scheduler]

        
        # ------------------------------- Loss Function ------------------------------ #

        self.Loss_Function=self.Loss_Function_Dict[self.Net['Loss_Function']]()
        


        # ---------------------------------------------------------------------------- #
        #                                    Dataset                                   #
        # ---------------------------------------------------------------------------- #

        
        self.DataSetType=self.DataSetConfig['Type']
        self.DataSet_Root=self.DataSetConfig['root']
        self.Dataset_Train_file=os.path.join(self.DataSet_Root,self.DataSetConfig['train_index_file'])
        self.Dataset_Val_file=os.path.join(self.DataSet_Root,self.DataSetConfig['val_index_file'])
        self.DefaultDataset=self.DataSetConfig['DefaultDataset']
        

        # --------------------------------- Transform -------------------------------- #
        # ---------------------------------------------------------------------------- #


        """
        Because the defalut detection network has transform flow 
        so the image list should include 3d tensors
        
        [
        [C, H, W],
        [C, H, W].....
        ]

        Target should be 
        list of dict :
        {
            boxes:      list of box tensor[n,4]                 (float32)
            masks:      list of segmentation mask points [n,n]  (float32)
            keypoints： list of key pointss[n,n]                (float32)
            labels:     list of index of label[n]               (int64)
        }
    
        For Default Detection:

        The transformations it perform are:
            - input normalization (mean subtraction and std division)
            - input / target resizing to match min_size / max_size

        It returns a ImageList for the inputs, and a List[Dict[Tensor]] for the targets
        
        """
        print('\n\n--------------------------------- Transform --------------------------------')
        self.Transform=self.DataSetConfig['Transform']
        functionlist=[list(i.keys())[0] for i in self.Transform]
        paralist=[list(i.values())[0] for i in self.Transform]
        
        
        
        
        self.transforms=GeneralTransform(self.TransformDict)
        self.target_transform=self.transforms
        # for i in range(len(functionlist)):

        
        #     print("-----Transform function :",functionlist[i]," para : ",paralist[i])

        #     if paralist[i]=="None":
        #         self.transforms.append(self.Transform_Class_Dict[functionlist[i]]())
        #         continue
        #     if type(paralist[i])==list:
        #         self.transforms.append(self.Transform_Class_Dict[functionlist[i]](*paralist[i]))
        #         continue
        #     # self.transforms.append(
        #     #     self.Transform_Function_Dict[functionlist[i]](paralist[i])
        #     # )

        # for i in range(len(functionlist)):
        #     print("-----Transform function :",functionlist[i]," para : ",paralist[i])
        #     if paralist[i]!="None": 
        #         self.transforms.append(
        #             self.Transform_Class_Dict[functionlist[i]](*paralist[i])
        #         )
        #     else:
        #         self.transforms.append(
        #             self.Transform_Class_Dict[functionlist[i]]()
        #         )
            





        # ---------------------------------------------------------------------------- #
        #                                    Config                                    #
        # ---------------------------------------------------------------------------- #
                
        self.DistributedDataParallel=self.Config['DistributedDataParallel']
        self.resume=self.Config['Resume']
        self.checkpoint=self.Config['checkpoint_path']
        self.MultiScale_Training=self.Config['multiscale_training']
        self.logdir=self.Config['logdir']
        self.devices=self.Config['devices']

        
        if self.devices=='GPU':
            self.usegpu=True
            self.gpu_id=self.Config['gpu_id']
            os.environ['CUDA_VISIBLE_DEVICES']=str(self.gpu_id)
            self.device = torch.device("cuda:"+str(self.gpu_id) if torch.cuda.is_available() else "cpu")
            print('--------------------------------- Transform --------------------------------\n\n-----Device:\n\n',self.device)
        
        if self.devices=='CPU':
            self.device=torch.device("cpu")


        self.download_pretrain_model=self.Config['down_pretrain_model']
        self.visualization=self.Config['visualization']
        self.worker_num=self.Config['worker_num']
        self.epochs=self.Config['epochs']
        self.aspect_ratio_factor=self.Config['group_factor']


        print("\n\n---------------------- Configure Class Init Successful ---------------------\n\n")











    # ---------------------------------------------------------------------------- #
    #                             Config Class Function                            #
    # ---------------------------------------------------------------------------- #

    def GenerateDefaultConfig(self,mode='detection'):
        print('Generate Default Config with mode :',mode)
    
    def configinfo(self):
        print('***** Already read Config file ,'+self.__configfile,'*****')
        print('***** Instance ID : ',self.InstanceID,'*****')
        print('***** Mission Type : ',self.MissionType,'*****')

    def Enviroment_Info(self):
        self.print_dict(self.__json)
        print('\n-------------------------------------------NVCC info:\n')
        os.system('nvcc -V')
        print('\n-------------------------------------------GPU info:\n')
        os.system('nvidia-smi')
        print('\n-------------------------------------------GPU info:\n')

    
    def print_dict(self,d,n=0):
        for k,v in d.items():
            # print ('\t'*n)
            if type(v)==type({}):
                print("%s : {" % k)
                self.print_dict(v,n+1)
            else:
                print("%s : %s" % (k,v))
        if n!=0:
            print('\t'*(n-1)+ '}')
