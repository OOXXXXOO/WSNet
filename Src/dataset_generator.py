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
        
        support_Mission={
            'Detection':['COCO2014','COCO2017','Pascal_VOC'],
            'Segmentation':['Cityscapes','COCO2014','COCO2017'],
            'InstenceSegmentation':['COCO2014','COCO2017'],   
            'Classification':['MINST','CIFAR10','CIFAR100','ImageNet'] 
            }
        
        """
        KeyPoint Detection , Image Caption in 2020
        """

        print('\n\n-----Dataset Generator Class init-----\n\n-----Support Mission:')
        self.print_dict(support_Mission)
        print('\n')

        
        
        #####Mission Type Checking
        assert self.MissionType in support_Mission.keys(),"Invalid MissionType"+self.MissionType
        
        #####DatasetType Checking
        assert self.DataSetType in support_Mission[self.MissionType],"Invalid DatasetSetType For this Mission"+self.DataSetType
        
      
        
        
        
        self.getitem_map={
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

        (KeyPoint,Image Caption in future     ==>  COCO2014,17)


        COCO Dataset File Structure:
        ---root
            |---annotations
                |---instances_train2014.json
                |---...
            |---train201x
                |---images.png
            |---val201x
                |---images.png

        Throughout the API 
        "ann"=annotation, 
        "cat"=category, and "img"=image.
        getAnnIdsGet ann ids that satisfy given filter conditions. 
        getCatIdsGet cat ids that satisfy given filter conditions. 
        getImgIdsGet img ids that satisfy given filter conditions. 
        loadAnnsLoad anns with the specified ids. 
        loadCatsLoad cats with the specified ids. 
        loadImgsLoad imgs with the specified ids. 
        loadResLoad algorithm results and create API for accessing them. 
        showAnnsDisplay the specified annotations.
            
        
        """
        print('*****Mode : ',self.MissionType,'-----start build dataset in :',self.DataSetType,'-----')
        print('*****DatasetRoot Dir',self.DataSet_Root,'*****')


        #####COCOFormat
        if self.DataSetType=='COCO2014' or self.DataSetType=='COCO2017':
            dataDir=self.DataSet_Root
            dataType=self.DataSetType[4:]
            if self.MissionType=='Detection' or self.MissionType=='InstenceSegmentation' or self.MissionType=='Segmentation':
                TrainAnnFile='{}/annotations/instances_{}.json'.format(dataDir,'train'+dataType)
                ValAnnFile='{}/annotations/instances_{}.json'.format(dataDir,'val'+dataType)
                
                print('\n\n*****Process Train anno file : ',TrainAnnFile)
                self.Traincoco=COCO(TrainAnnFile)
                
                print('\n\n*****Process Val anno file',ValAnnFile)
                self.Valcoco=COCO(ValAnnFile)

                self.COCOimageslist=self.Traincoco.getImgIds()
                self.COCOannalist=self.Traincoco.getAnnIds()
                self.COCOcatlist=self.Traincoco.getCatIds()


                cats = self.Traincoco.loadCats(self.Traincoco.getCatIds())
                self.nms=[cat['name'] for cat in cats]
                print('\n\n*****COCO categories: \n{}\n'.format(' '.join(self.nms)))

                self.supernms = set([cat['supercategory'] for cat in cats])
                print('\n\n*****COCO supercategories: \n{}'.format(' '.join(self.supernms)))
                self.cats=cats
                self.DataSetProcessDone=True
            if self.MissionType=='Caption':
                print('Not Support Now~')
                exit(0)
            
        #####CityscapesFormat
        if self.DataSetType=='Cityscapes':
            self.DataSetProcessDone=True
     
        #####MINSTFormat
        if self.DataSetType=='MINST':
            self.DataSetProcessDone=True
    
        #####PascalVocFormat
        if self.DataSetType=='Pascal_VOC':
            self.DataSetProcessDone=True
        
        #####ImageNetFormat
        if self.DataSetType=='ImageNet':
            self.DataSetProcessDone=True
        







    def __getCIFAR(self,index):
        pass
        
    def __getitemCOCO(self,index):
        image=self.Traincoco.imgs[self.COCOimageslist[index]]
        anns=self.Traincoco.anns[self.COCOannalist[index]]
        cat=self.Traincoco.cats[self.COCOcatlist[index]]
        
        """
        food vehicle indoor electronic person animal sports furniture kitchen appliance accessory outdoor
        image:
        {'license': 1, 'file_name': 'COCO_train2014_000000222016.jpg', 
        'coco_url': 'http://images.cocodataset.org/train2014/COCO_train2014_000000222016.jpg', 
        'height': 640, 'width': 480, 'date_captured': '2013-11-14 16:37:59',
         'flickr_url': 'http://farm2.staticflickr.com/1431/1118526611_09172475e5_z.jpg', 
         'id': 222016} 
        anns:
        {'segmentation': 
        [[292.92, 25.6, 288.4, 41.42, 289.91, 61.75, 289.16, 180.72, 314.76, 289.16, 347.14, 324.55, 356.17, 334.34, 384.79, 328.31, 391.57, 327.56, 394.58, 326.05, 402.86, 328.31, 432.98, 337.35, 429.22, 343.37, 438.25, 349.4, 454.82, 343.37, 458.58, 330.57,
         460.09, 333.58, 487.2, 323.8, 499.25, 312.5, 500.0, 300.45, 500.0, 281.63, 499.25, 266.57, 499.25, 236.45, 498.49, 206.33, 498.49,
          185.24, 491.72, 168.67, 448.04, 83.58, 418.67, 48.19, 399.1, 28.61, 396.08, 24.85, 378.77, 36.14, 376.51, 36.9, 354.67, 30.12, 338.1,
           32.38, 340.36, 36.9, 330.57, 45.18, 329.82, 48.95, 327.56, 48.95, 326.05, 44.43, 324.55, 30.87, 320.78, 19.58, 316.27, 18.07, 307.23, 18.07, 302.71, 23.34, 298.19, 24.1, 295.18, 24.85]],
            'area': 53535.29024999999, 
            'iscrowd': 0, 'image_id': 142589, 
            'bbox': [288.4, 18.07, 211.6, 331.33],
             'category_id': 58, 'id': 93} 
        cat:
        {'supercategory': 'vehicle', 'id': 3, 'name': 'car'}
        """

        return image,anns,cat
    
    def __getitemPascal(self,index):
        pass

    def __getitemCitys(self,index):
        pass
    
    def __getitemImNets(self,index):
        pass

    def __getitemMINST(self,index):
        pass

    def __getitem__(self,index):
        assert self.DataSetProcessDone,"Invalid Dataset Object"
        return self.getitem_map[self.DataSetType](index)



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
    COCO2014[2]















if __name__ == '__main__':
    main()
    