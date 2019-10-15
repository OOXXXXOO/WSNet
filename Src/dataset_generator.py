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
        
        #######################################
        #Decide the getitem & len can be use
        self.DataSetProcessDone=False
        
        support_Mode={
            'Detection':
            ['COCO2014','COCO2017','Pascal_VOC'],
            
            'Segmentation':
            ['Cityscapes','COCO2014','COCO2017'],
            
            'InstenceSegmentation':
            ['COCO2014','COCO2017'],   
            
            'Classification':
            ['MINST','CIFAR10','CIFAR100','ImageNet'] 
            }
        """
        KeyPoint Detection , Image Caption
        in 2020
        """
        
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
        # assert self.DataSetType in support_Dataset,"Invalid Dataset Type : "+self.DataSetType
        # assert self.MissionType in support_Mode,"Invalid Mission Type : "+self.MissonType 
        # change strategy of type checking
        getitem_map={
            "COCO2014":self.__getitemCOCO,
            "COCO2017":self.__getitemCOCO,
            "CitysCapes":self.__getitemCitys,
            "MINST":self.__getitemMINST,
            "CIFAR10":self.__getCIFAR,
            "CIFAR100":self.__getCIFAR,
            "PascalVOC":self.__getitemPascal,
            "ImageNet":self.__getitemImNets
        }


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
        Mission Type:
        
        Detection
        Segmentation
        InstenceSegmentation
        Classification

        DatasetName:

        Classification          ==>  MINST,CIFAR,ImageNet
        Detection               ==>  COCO2014,COCO2017,Pascal_VOC
        InstenceSegmentation    ==>  COCO2014,COCO2017
        Segmentation            ==>  Cityscapes,COCO2014,COCO2017

        (KeyPoint in future     ==>  COCO2014,17)
        
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
            self.Valcoco=COCO(ValAnnFile)

            cats = self.Traincoco.loadCats(self.Traincoco.getCatIds())
            self.nms=[cat['name'] for cat in cats]
            print('*****COCO categories: \n{}\n'.format(' '.join(self.nms)))

            self.supernms = set([cat['supercategory'] for cat in cats])
            print('*****COCO supercategories: \n{}'.format(' '.join(self.supernms)))
            self.cats=cats
            self.DataSetProcessDone=True
        
        
        if self.DataSetType=='Cityscapes':
            self.DataSetProcessDone=True
     
        
        if self.DataSetType=='MINST':
            self.DataSetProcessDone=True
    
        
        if self.DataSetType=='Pascal_VOC':
            self.DataSetProcessDone=True
        
    def __getCIFAR():
        pass
        
    def __getitemCOCO():
        pass
    
    def __getitemPascal():
        pass

    def __getitemCitys():
        pass
    
    def __getitemImNets():
        pass

    def __getitemMINST():
        pass

    def __getitem__(self,index):
        assert self.DataSetProcessDone,"Invalid Dataset Object"
        return self.getitem_map[self.DataSetType]()



    def __len__(self):    
        assert self.DataSetProcessDone,"Invalid Dataset Object"
        
    





    def COCO_Demo_Visulization(self):

        # get all images containing given categories, select one at random
        catIds = self.Traincoco.getCatIds(catNms=['person','dog','skateboard']);
        imgIds = self.Traincoco.getImgIds(catIds=catIds );
        img = self.Traincoco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
        I = io.imread(img['coco_url'])

        plt.imshow(I); plt.axis('off')
        annIds = self.Traincoco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = self.Traincoco.loadAnns(annIds)
        print('anns ', anns)
        """
        Anns Format :
        {   
            'segmentation': [[463.43, 154.2, 468.15, 156.56, 472.87, 158.4, 474.97, 157.09, 
        477.07, 155.25, 478.91, 154.2, 479.17, 153.94, 487.57, 156.56, 485.99, 159.97,
            479.17, 158.92, 476.29, 159.45, 469.2, 162.33, 467.36, 166.01, 463.43, 164.69,
            461.07, 164.17, 458.7, 163.38, 454.77, 164.17, 451.62, 164.17, 448.73, 162.6, 
            450.31, 159.18, 449.0, 158.66, 442.17, 156.04, 446.63, 155.51, 450.57, 154.99,
            453.46, 154.72]],
            'area': 267.6581999999997, 
            'iscrowd': 0, 
            'image_id': 568187, 
            'bbox': [442.17, 153.94, 45.4, 12.07], 
            'category_id': 41, 'id': 639525
        }

        """
        self.Traincoco.showAnns(anns)
        plt.show()


    def DatasetInfo(self):
        print('dataset class on pytorch version :',torch.__version__)












def main():
    COCO2014=DatasetGenerator()
    COCO2014.DefaultDataset()















if __name__ == '__main__':
    main()
    