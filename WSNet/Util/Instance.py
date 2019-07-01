import torchvision.datasets as dataset
import torch.nn as nn
import torch.utils as util
import os
import sys
sys.path.append('../')

from config import *
from datasets import *
from network import *

MissionType = ['Detection', 'Segmentation', 'Mask']
Motto='|||||||-----Main Process of Fast Train Net-----|||||||\n\n'



class Instance(cfg,
               custom_dataloader,
               dataset_generator,
               networkgenerator):

    def __init__(self,MissionIndex=0,jsondir='./config_json'):
        super(Instance,self).__init__(cfg,custom_dataloader,dataset_generator,networkgenerator)
        
        """
                                      |——template-config-generator——>——>|
                        |——Config-----|——readconfig<————————————————————|  
                        |     ^       |——configure instance             |  
                        |     |                                         |               
        Instance[MODE]——|-->Dataset-->|——training-array generator       |  
                        |             |——training-to DataLoader         |  
                        |                                               |          
                        |——Network----|——readconfig<————————————————————|  
                                      |——Network Generator
                                      |——Network Process——————>——————————>Train/Val/Test

        MODE=[Segmentation,Detection,Mask]
        """
        Mission=MissionType[MissionIndex]
        BackBoneAvaliable=True
        BackBoneType=['Resnet','DenseNet','Xception']
        DectionType=['FasterRCNN','CascadeRCNN','MaskRCNN','YoloV3']
        SegmentationType=['Deeplabv3plus','SegNet']
        print(Motto)
        print('*******************Instance mode : ',Mission,' *******************')
        


    def fileprocess(self,path):
        assert path.type()!=str,'path format error'
        if not os.path.exists(path):
            os.makedirs(path)
            



    def NetInit(self):
        print('****NetInit****')

    def BackBoneInit(self):
        print('****BackBone****')

    def ReadConfig(self):
        print('****ConfigFile Read****')



def main():
    ins=Instance()
    




if __name__ == '__main__':
    main()
    