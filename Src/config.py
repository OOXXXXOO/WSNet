# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    config.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/28 11:45:40 by winshare          #+#    #+#              #
#    Updated: 2020/05/28 15:01:24 by winshare         ###   ########.fr        #
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
import os
import json
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models
import torchvision.datasets as dataset

# ---------------------------- Official Reference ---------------------------- #
from Src.description import Description


# ------------------------------ Local Reference ----------------------------- #

class CFG(Description):
    def __init__(self):

        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #
        Description.__init__(self)
        for i in range(5):
            print("#####------------------------------------------------------------------#####")
        print("#####-------------------------<===== WSNET =====>----------------------#####")
        for i in range(5):
            print("#####------------------------------------------------------------------#####")

     
        print("\n\n# -----Decode Config File :",self.configfile,"-----#")


  

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

        print('\n\n# ---------------------------------- config ---------------------------------- #')
       
        print("# ------------------------------ NETWORK CONFIG ------------------------------ #")
        self.print_dict(self.Net)
        print("# ------------------------------ NETWORK CONFIG ------------------------------ #")

        print("# ------------------------------ DATASET CONFIG ------------------------------ #")
        self.print_dict(self.DataSetConfig)
        print("# ------------------------------ DATASET CONFIG ------------------------------ #")

        print("# ------------------------------ GENERAL CONFIG ------------------------------ #")
        self.print_dict(self.Config)
        print("# ------------------------------ GENERAL CONFIG ------------------------------ #")

        print('# ---------------------------------- config ---------------------------------- #')


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


        


        self.NetType=self.Net["NetType"]
        

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
        self.class_num=self.Net['class_num']
        
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
        self.NPY=self.DataSetConfig["NPY"]
        if os.path.exists(self.NPY):
            self.NPY_Data=np.load(self.NPY,allow_pickle=True)
    
        
        self.SFT_Enable=self.DataSetConfig["SFT_Enable"]

        # --------------------------------- Transform (Aborted)------------------------#
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
        # print('\n\n--------------------------------- Transform --------------------------------')
        # self.TransformDict=self.DataSetConfig['Transform']
        # functionlist=[list(i.keys())[0] for i in self.TransformDict]
        # paralist=[list(i.values())[0] for i in self.TransformDict]
        
        
        
        
        # self.transforms=GeneralTransform(self.TransformDict)
  




        # ---------------------------------------------------------------------------- #
        #                                    Config                                    #
        # ---------------------------------------------------------------------------- #
                
        self.DistributedDataParallel=self.Config['DistributedDataParallel']
        self.resume=self.Config['Resume']
        self.checkpoint=self.Config['checkpoint_path']
        self.MultiScale_Training=self.Config['multiscale_training']
        self.logdir=self.Config['logdir']
        self.devices=self.Config['devices']
        self.pre_estimation=self.Config['pre_estimation']

        if not os.path.exists(self.checkpoint):
            os.makedirs(self.checkpoint)


        
        if self.devices=='GPU':
            self.usegpu=True
            self.gpu_id=self.Config['gpu_id']
            # os.environ['CUDA_VISIBLE_DEVICES']=str(self.gpu_id)
            self.device = torch.device("cuda:"+str(self.gpu_id) if torch.cuda.is_available() else "cpu")
            print('#-----Device:\n',self.device)
        
        if self.devices=='CPU':
            self.device=torch.device("cpu")


        self.download_pretrain_model=self.Config['down_pretrain_model']
        self.visualization=self.Config['visualization']
        self.worker_num=self.Config['worker_num']
        self.epochs=self.Config['epochs']
        self.aspect_ratio_factor=self.Config['group_factor']

        print("# ---------------------------------------------------------------------------- #")
        print("#                        Configure Class Init Successful                       #")
        print("# ---------------------------------------------------------------------------- #")
        self.Enviroment_Info()








        # ---------------------------------------------------------------------------- #
        #                                  EXTRA PART                                  #
        # ---------------------------------------------------------------------------- #







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
        print("\n\n# --------------------------------- NVCC INFO -------------------------------- #\n\n")
        os.system('nvcc -V')
        print("\n\n# --------------------------------- NVCC INFO -------------------------------- #\n\n")
        
        print("\n\n# --------------------------------- GPU INFO --------------------------------- #")
        os.system('nvidia-smi')
        print("# --------------------------------- GPU INFO --------------------------------- #\n\n")
    
    def print_dict(self,d,n=0):
        length=74
        for k,v in d.items():
            # print ('\t'*n)
            if type(v)==type({}):
                print("%s : {" % k)
                self.print_dict(v,n+1)
            else:
                strl=len(str(k))+len(str(v))
                space=length-strl
                print("# %s : %s" % (k,v)+" "*space+"#")
        if n!=0:
            print('\t'*(n-1)+ '}')
