import pandas as pd
import os
import json
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as T
class cfg():
    def __init__(self,configfile='Src/config/config_template.json'):
        print('\n\n-----Configure Generator Class Init -----\n\n')
        ##File Level
        self.__defaultconfig=configfile
        self.__json=json.load(open(self.__defaultconfig,'r'))
        self.usegpu=False


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
        # self.NetType=self.Net['NetType']
        self.DefaultNetwork=self.Net["DefaultNetwork"]
        
    
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
        ###########
        #Transform
        self.Transform_dict={
                # "CenterCrop":T.CenterCrop(),
                # "ColorJitter":T.ColorJitter(),
                # "adjust_brightness":T.functional.adjust_brightness(),
                # "adjust_contrast":T.functional.adjust_contrast(),
                # "adjust_gamma":T.functional.adjust_gamma(),
                # "adjust_hue":T.functional.adjust_hue(),
                # "adjust_saturation":T.functional.adjust_saturation(),
                # "affine":T.functional.affine(),
                # "crop":T.functional.crop(),
                # "erase":T.functional.erase(),
                # "five_crop":T.functional.five_crop(),
                # "hflip":T.functional.hflip(),
                # "normalize":T.functional.normalize(),
                # "pad":T.functional.pad(),
                # "perspective":T.functional.perspective(),
                # "resize":T.functional.resize(),
                # "resized_crop":T.functional.resized_crop(),
                # "rotate":T.functional.rotate(),
                # "ten_crop":T.functional.ten_crop(),
                # "to_grayscale":T.functional.to_grayscale(),
                # "to_pil_image":T.functional.to_pil_image(),
                # "to_tensor":T.functional.to_tensor(),
                # "vflip":T.functional.vflip(),
                # "Grayscale":T.Grayscale(),
                # "Lambda":T.Lambda(),
                "Normalize":T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                # "Pad":T.Pad(),
                # "RandomAffine":T.RandomAffine(),
                # "RandomApply":T.RandomApply(),
                # "RandomChoice":T.RandomChoice(),
                # "RandomCrop":T.RandomCrop(),
                "RandomErasing":T.RandomErasing(),
                # "RandomGrayscale":T.RandomGrayscale(),
                "RandomHorizontalFlip":T.RandomHorizontalFlip(0.5),
                # "RandomOrder":T.RandomOrder(),
                # "RandomPerspective":T.RandomPerspective(),
                # "RandomResizedCrop":T.RandomResizedCrop(),
                # "RandomRotation":T.RandomRotation(),
                # "RandomSizedCrop":T.RandomSizedCrop(),
                "RandomVerticalFlip":T.RandomVerticalFlip(),
                # "Resize":T.Resize(),
                # "Scale":T.Scale(),
                # "TenCrop":T.TenCrop(),
                "ToPILImage":T.ToPILImage(),
                "ToTensor":T.ToTensor(),
        }
        """
        Transform function dict use name string to map Transfrom function &
        use T.Compose() to control the transform process
        """
        self.Transform=self.DataSetConfig['Transform']
        self.Transform=[self.Transform_dict[i] for i in self.Transform]
        self.image_transforms=T.Compose(self.Transform)
        print(self.image_transforms)
        








        #---Config
        self.checkpoint=self.Config['checkpoint_path']
        self.MultiScale_Training=self.Config['multiscale_training']
        self.logdir=self.Config['logdir']
        self.devices=self.Config['devices']
        if self.devices=='GPU':
            self.usegpu=True
            self.gpu_id=self.Config['gpu_id']
            os.environ['CUDA_VISIBLE_DEVICES']=str(self.gpu_id)
            self.device = torch.device("cuda:"+str(self.gpu_id) if torch.cuda.is_available() else "cpu")
            print('train device on :',self.device)


        self.download_pretrain_model=self.Config['down_pretrain_model']
        self.visualization=self.Config['visualization']
        self.worker_num=self.Config['worker_num']
        self.epochs=self.Config['epochs']



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

    