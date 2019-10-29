import pandas as pd
import os
import json
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
class cfg():
    def __init__(self,configfile='Src/config/config_template.json'):
        print('\n\n-----Configure Generator Class Init -----\n\n')
        ##File Level
        self.__defaultconfig=configfile
        self.__json=json.load(open(self.__defaultconfig,'r'))


        ##First Content Level
        self.MissionType=self.__json['MissionType']
        self.InstanceID=self.__json['instance_id']
        self.Content=self.__json['content']

        print('***** Already read Config file ,'+self.__defaultconfig,'*****')

        print('***** Instance ID : ',self.InstanceID,'*****')

        print('***** Mission Type : ',self.MissionType,'*****')

        
        
        #Second Level
        self.Net=self.Content['Net']
        self.DataSetConfig=self.Content['Dataset']
        self.Config=self.Content['Config']

        print('***** Network Config : ',self.Net)
        print('***** Dataset Config : ',self.DataSetConfig)
        print('***** General Config : ',self.Config)




        #Third Level
        
        #---NET
        self.NetType=self.Net['NetType']
        self.BatchSize=self.Net['BatchSize']
        if self.Net['BackBone']=='None':
            self.BackBone=None
        else:
            self.BackBone=self.Net['BackBone']
        #####Optimizer
        OptimDict={

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
        self.Optimzer=OptimDict[self.Net['Optimizer']]

        #####Loss Function
        self.Loss_Function_dict={
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
        self.Loss_Function=self.Loss_Function_dict[self.Net['Loss_Function']]



        #---Dataset
        self.DataSetType=self.DataSetConfig['Type']
        self.DataSet_Root=self.DataSetConfig['root']
        self.Dataset_Train_file=os.path.join(self.DataSet_Root,self.DataSetConfig['train_index_file'])
        self.Dataset_Val_file=os.path.join(self.DataSet_Root,self.DataSetConfig['val_index_file'])

        #---Config
        self.checkpoint=self.Config['checkpoint_path']
        self.MultiScale_Training=self.Config['multiscale_training']
        self.logdir=self.Config['logdir']
        self.gpu_id=self.Config['gpu_id']
        self.download_pretrain_model=self.Config['down_pretrain_model']
        self.visualization=self.Config['visualization']



    def GenerateDefaultConfig(self,mode='detection'):
        print('Generate Default Config with mode :',mode)
        # print(self.__json)



















    def print_dict(self,d,n=0):
        for k,v in d.items():
            print ('\t'*n)
            if type(v)==type({}):
                print("%s : {" % k)
                self.print_dict(v,n+1)
            else:
                print("%s : %s" % (k,v))
        if n!=0:
            print('\t'*(n-1)+ '}')

    def Enviroment_Info(self):
        self.print_dict(self.__json)

        print('\n-------------------------------------------NVCC info:\n')
        os.system('nvcc -V')
        print('\n-------------------------------------------GPU info:\n')
        os.system('nvidia-smi')
        print('\n-------------------------------------------GPU info:\n')

    