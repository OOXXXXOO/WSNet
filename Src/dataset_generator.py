import torchvision.datasets as dataset
import torch

class DatasetGenerator():
    def __init__(self):
        print('\n\n-----Dataset Generator init-----\n\n')




    def CustomDataset(self,root,Ratio=0.7,mode='Detection'):
        """
        mode:
        
        Detection
        Segmentation
        InstenceSegmentation
        Classification

        root file structure:
        

        """
        print('root in : ',mode,'-----start build',DatasetName,'-----')








    def DefaultDataset(self,DatasetName=Cityscapes,mode='Detection'):
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