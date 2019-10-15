import torchvision.datasets as dataset
import torch
from config_generator import *
from pycocotools.coco import COCO
import os

##################################################
#For Visualization
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
##################################################


BusinessCOCODatasetRoot='/media/winshare/98CA9EE0CA9EB9C8/COCO_Dataset/'

class DatasetGenerator(cfg,COCO):
    def __init__(self):
        super(DatasetGenerator,self).__init__()
        super(cfg,self).__init__()
        
        support_Mode=[
        'Detection',
        'Segmentation',
        'InstenceSegmentation',
        'Classification' 
        ]
        
        support_Dataset=[
            "COCO2014",
            "COCO2017",
            "CitysCapes",
            "MINST",
            "CIFAR10",
            "CIFAR100",
            "PascalVOC",
            "ImageNet"
        ]

        print('\n\n-----Dataset Generator Class init-----\n\n')
        assert self.DataSetType in support_Dataset,"Invalid Dataset Type : "+self.DataSetType
        assert self.MissionType in support_Mode,"Invalid Mission Type : "+self.MissonType 




    def CustomDataset(self,root='./',Ratio=0.7,mode='Detection'):
        """
        mode:
        
        Detection
        Segmentation
        InstenceSegmentation
        Classification

        root file structure:
        

        """
        print('Custom Root in ',self.root,'Mode : ',self.MissionType,'-----start custom build-----')



    def Custom2COCO():
        """
        Support for transfer the CustomDataset to COCO format
        """
        print('Start Transfrom Custom Dataset 2 COCO 201x Dataset Format')
        pass




    def DefaultDataset(self):
        """
        mode:
        
        Detection
        Segmentation
        InstenceSegmentation
        Classification

        DatasetName:

        Classification          ==>  MINST,CIFAR,ImageNet
        Detection               ==>  COCO2014,COCO2017,Pascal_VOC
        InstenceSegmentation    ==>  COCO2014,COCO2017
        Segmentation            ==>  Cityscapes

        (KeyPoint in future      ==>  COCO2014,17)
        
        """
        print('*****Mode : ',self.MissionType,'-----start build dataset in :',self.DataSetType,'-----')
        print('*****DatasetRoot Dir',self.DataSet_Root,'*****')
        if self.DataSetType=='COCO2014' or self.DataSetType=='COCO2017':
            dataDir=self.DataSet_Root
            dataType=self.DataSetType[4:]
            TrainAnnFile='{}/annotations/instances_{}.json'.format(dataDir,'train'+dataType)
            ValAnnFile='{}/annotations/instances_{}.json'.format(dataDir,'val'+dataType)
            
            print('\n\n*****Process Train anno file : ',TrainAnnFile)
            self.Traincoco=COCO(TrainAnnFile)
            
            print('\n\n*****Process Val anno file',ValAnnFile)
            # self.Valcoco=COCO(ValAnnFile)

            cats = self.Traincoco.loadCats(self.Traincoco.getCatIds())
            self.nms=[cat['name'] for cat in cats]
            print('COCO categories: \n{}\n'.format(' '.join(self.nms)))

            self.supernms = set([cat['supercategory'] for cat in cats])
            print('COCO supercategories: \n{}'.format(' '.join(self.supernms)))
            self.cats=cats

            # get all images containing given categories, select one at random
            catIds = self.Traincoco.getCatIds(catNms=['person','dog','skateboard']);
            imgIds = self.Traincoco.getImgIds(catIds=catIds );
            img = self.Traincoco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
            I = io.imread(img['coco_url'])

            plt.imshow(I); plt.axis('off')
            annIds = self.Traincoco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
            anns = self.Traincoco.loadAnns(annIds)
            print('anns ', anns)
            self.Traincoco.showAnns(anns)
            plt.show()



    def DatasetInfo(self):
        print('dataset class on pytorch version :',torch.__version__)
        print('')












def main():
    Generator=DatasetGenerator()
    Generator.DefaultDataset()















if __name__ == '__main__':
    main()
    