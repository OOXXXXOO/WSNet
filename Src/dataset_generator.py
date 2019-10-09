import torchvision.datasets as dataset
import torch
from config_generator import *
from pycocotools.coco import COCO

BusinessCOCODatasetRoot='/media/winshare/98CA9EE0CA9EB9C8/COCO_Dataset/annotations_trainval2014'

class DatasetGenerator(cfg,COCO):
    def __init__(self):
        super(DatasetGenerator,self).__init__()
        super(cfg,self).__init__()
        
        print('\n\n-----Dataset Generator init-----\n\n')




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
        pass




    def DefaultDataset(self,DatasetName='Cityscapes',mode='Detection'):
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

        """
        print('Mode : ',mode,'-----start build',DatasetName,'-----')






    def DatasetInfo(self):
        print('dataset class on pytorch version :',torch.__version__)
        print('')












def main():

    dataType='val2014'
    annFile='{}/annotations/instances_{}.json'.format(BusinessCOCODatasetRoot,dataType)
    dataset=COCO(annFile)













if __name__ == '__main__':
    main()
    