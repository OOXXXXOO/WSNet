import torchvision.datasets as dataset
import torch.nn as nn
import torch.utils as util
import os

MissionType = ['Detection', 'Segmentation', 'Mask']


class Instance(object):
    def __init__(self,MissionIndex=0):
        """
        Session instance
        """
        Mission=MissionType[MissonIndex]
        BackBoneType=['Resnet','DenseNet','Xception']
        DectionType=['FasterRCNN','CascadeRCNN','MaskRCNN','YoloV3']
        SegmentationType=['Deeplabv3plus','SegNet']

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


