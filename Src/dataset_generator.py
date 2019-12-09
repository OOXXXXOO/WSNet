# Copyright 2019 Winshare
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


# -*- coding: utf-8 -*-
# @Author: Winshare
# @Date:   2019-12-02 17:44:47
# @Last Modified by:   Winshare
# @Last Modified time: 2019-12-05 13:59:27
# --------------------------------- Winshare --------------------------------- #



import torchvision.transforms as T
from torch.utils.data import Dataset
import torch
import config_generator.cfg as cfg 
from pycocotools.coco import COCO
import os


# ----------------------------- For Visualization ---------------------------- #

import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ------------------------------- For Transfrom ------------------------------ #

BusinessCOCODatasetRoot='/media/winshare/98CA9EE0CA9EB9C8/COCO_Dataset'




class DatasetGenerator(cfg,COCO,Dataset):

    def __init__(self,UseInternet=False,transforms=None):
        """
        UseInterNet=True  => load images by url from internet
        UseInterNet=False => load images by local path
        """
    

        print('\n\n-----Dataset Generator Class init-----\n\n:')
        
        cfg.__init__(self)
        super(DatasetGenerator,self).__init__()
        self.UseInternet=UseInternet
        self.DataSetProcessDone=False
        self.transforms=transforms
        self.dataset=self.datasets_function_dict[DataSetType]



            
        # ---------------------------------------------------------------------------- #
        #                                 Function Dict                                #
        # ---------------------------------------------------------------------------- #

        # 
        
        # ----------------------------- Getitem Function ----------------------------- #






        self.getitem_map={
            "MINST":self.__getitemMINST,
            # "FashionMINST":self.__getitemFMINST,
            # "KMINST":self.__getitemKMINST,
            # "EMINST":self.__getitemEMINST,
            # "FakeData":self.__getitemFakeData,
            "CocoCaptions":self.__getitemCocoCaption,
            "CocoDetection":self.__getitemCocoDetection,
            # "LSUN":self.__getitemLSUN,
            # "ImageFolder":self.__getitemImgFolder,
            # "DatasetFolder":self.__getitemSetFolder,
            "ImageNet":self.__getitemImageNet,
            "CIFAR10":self.__getitemCIFAR10,
            "CIFAR100":self.__getitemCIFAR100,
            "STL10":self.__getitemSTL10,
            # "SVHN":dataset.SVHN,
            # "PhotoTour":dataset.PhotoTour,
            # "SBU":dataset.SBU,
            # "Flickr30k":dataset.Flickr30k,
            "VOC_Detection":self.__getitemVOCDetection,
            "VOC_Segmentation":self.__getitemVOCSegmentation,
            "Cityscapes":self.__getitemCitysCapes
            # "SBD":dataset.SBDataset,
            # "USPS":dataset.USPS,
            # "Kinetics-400":dataset.Kinetics400,
            # "HMDB51":dataset.HMDB51,
            # "UCF101":dataset.UCF101
        }


        # --------------------------- Mission Type Checking -------------------------- #

        assert self.MissionType in support_Mission.keys(),"Invalid MissionType"+self.MissionType
        
        # --------------------------- DatasetType Checking --------------------------- #

        assert self.DataSetType in support_Mission[self.MissionType],"Invalid DatasetSetType For this Mission"+self.DataSetType
                
        # ---------------------------------------------------------------------------- #
        #                                   Transform                                  #
        # ---------------------------------------------------------------------------- #
        """
        Tips:
        0,完成Totensor不带RandomFlip版本
        1,改写Compose基类能够容纳Target对象
        2,通过改写Functional中的方法,来改写对象
        """


    def CustomDataset(self,root='./',Ratio=0.7,mode='Detection'):
        """
        mode:
        
        Detection
        Segmentations
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




    def DefaultDatasetFunction(self,Mode='train'):
        """
        Mode Type
        
        train
        test
        val

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
        Caption                 ==>  COCO2014,COCO2017

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


            
        
        """
        print('*****Mode : ',self.MissionType,'-----start build dataset in :',self.DataSetType,'-----')
        print('*****DatasetRoot Dir',self.DataSet_Root,'*****')
        assert Mode in ['train','val','test'],"Invalide DataSet Mode"
        self.Mode=Mode








        # ---------------------------------------------------------------------------- #
        #                                     abort                                    #
        # ---------------------------------------------------------------------------- #

        ####COCOFormat
        # if self.DataSetType=='COCO2014' or self.DataSetType=='COCO2017':
            # if Mode=='train':
                # self.coco = COCO(self.Dataset_Train_file)
                # self.ids = list(sorted(self.coco.imgs.keys()))
                # self.datasetroot=os.path.join(self.DataSet_Root,'train'+self.DataSetType[4:])
                # print('Train Data Folder Root :',self.datasetroot)
            # if Mode=='val':
                # self.coco = COCO(self.Dataset_Val_file)
                # self.ids = list(sorted(self.coco.imgs.keys()))
                # self.datasetroot=os.path.join(self.DataSet_Root,'val'+self.DataSetType[4:])
                # print('Val Data Folder Root :',self.datasetroot)
            # self.DataSetProcessDone=True
# 
# 
        ####CityscapesFormat
        # if self.DataSetType=='Cityscapes':
            # self.DataSetProcessDone=True
    #  
        ####MINSTFormat
        # if self.DataSetType=='MINST':
            # self.DataSetProcessDone=True
    # 
        ####PascalVocFormat
        # if self.DataSetType=='Pascal_VOC':
            # self.DataSetProcessDone=True
        # 
        ####ImageNetFormat
        # if self.DataSetType=='ImageNet':
            # self.DataSetProcessDone=True
        




    def __getitemCocoDetection(self,index):
        pass



    def __getCIFAR(self,index):
        pass
        
    def __getitemCOCO(self,index):
        """
        Args:
        index (int): Index
        Support for Caption & Detection & Segmentation
    
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        if self.MissionType=='Detection':
            coco = self.coco
            img_id = self.ids[index]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)
            path = coco.loadImgs(img_id)[0]['file_name']
            img = Image.open(os.path.join(self.datasetroot, path)).convert('RGB')
            if self.transforms is not None:
                img,target=self.transforms(img,target)
            

        if self.MissionType=='Caption':
            coco = self.coco
            img_id = self.ids[index]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            target = [ann['caption'] for ann in anns]
            path = coco.loadImgs(img_id)[0]['file_name']
            img = Image.open(os.path.join(self.datasetroot, path)).convert('RGB')
            if self.image_transforms is not None:
                img,target=self.transforms(img,target)
        if self.MissionType=='Segmentation':
            # -------------------------- segmentation_label2mask ------------------------- #
            pass
        if self.MissionType=='Instance':
            # --------------------------- instance segmentation -------------------------- #
            pass

        return img, target


    
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
        return len(self.ids)
        
    
    




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



    def collate_fn(self,batch):
        return tuple(zip(*batch))





    def DatasetInfo(self):
        print('dataset class on pytorch version :',torch.__version__)












def main():
    COCO2014=DatasetGenerator()
    COCO2014.DefaultDataset()
    print(COCO2014[20])















if __name__ == '__main__':
    main()
    