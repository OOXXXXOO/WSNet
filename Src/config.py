# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    config.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <winshare@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/06/22 16:42:50 by tanwenxuan        #+#    #+#              #
#    Updated: 2020/06/23 16:21:28 by winshare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

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



from Src.reference import reference

# ------------------------------ local reference ----------------------------- #

import json
import os






class cfg(reference):
    def __init__(self):
        reference.__init__(self)    
        print("# ---------------------------------------------------------------------------- #")
        print("#                           config decoder init start                          #")
        print("# ---------------------------------------------------------------------------- #")
        
        print("# ===== Decode Config \33[1;32m%s\33[0m"%self.config) # Init on class model
        self.__configfile=self.config
        self.__json=json.load(open(self.__configfile,'r'))


        # -------------------------------- First Level ------------------------------- #

        self.MissionType=self.__json['MissionType']
        self.InstanceID=self.__json['instance_id']
        self.Content=self.__json['content']


        # ------------------------------- Second Level ------------------------------- #
        self.Net=self.Content['Net']
        self.DataSetConfig=self.Content['Dataset']
        self.Config=self.Content['Config']

        self.debug=self.Config["debug"]
        
        print('# ---------------------------------- config ---------------------------------- #')
       
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
        self.DefaultNetwork=self.Net["DefaultNetwork"]
        self.BatchSize=self.Net['BatchSize']
        if self.Net['BackBone']=='None':
            self.BackBoneName=None
        else:
            self.BackBoneName=self.Net['BackBone']
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









        print("# ---------------------------------------------------------------------------- #")
        print("#                           config decoder init done                           #")
        print("# ---------------------------------------------------------------------------- #")




    def print_dict(self,d,n=0):
        """
        charactorize dict output
        """
        length=74
        for k,v in d.items():
            # print ('\t'*n)
            if type(v)==type({}):
                print("%s : {" % k)
                self.print_dict(v,n+1)
            else:
                strl=len(str(k))+len(str(v))
                space=length-strl
                print("# %s : \33[1;32m%s\33[0m" % (k,v)+" "*space+"#")
        if n!=0:
            print('\t'*(n-1)+ '}')






















