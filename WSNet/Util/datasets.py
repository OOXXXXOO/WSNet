import cv2
import torch.nn as nn
import os
import argparse

import numpy
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import PIL.Image as Image



"""template
class trainset(Dataset):
    def __init__(self,loader=default_loader):
        self.images=imagelist[:6000]
        self.labels=labellist[:6000]
        self.loader=loader
    def __getitem__(self, index):

        fn=self.images[index]
        img=self.loader(fn)
        labels=self.labels[index]
        labels=np.array(labels)
        labels=torch.from_numpy(labels)
        return img,labels

    def __len__(self):
        return len(self.images)
"""


class custom_dataloader(Dataset):
    def __init__(self,data_list=[],label_list=[],trainval_ratio=0.9,image_size=512):
        print('custom_dataloader module init done.')
        super(custom_dataloader,self).__init__()
        """
        data_list
        label_list
        trainval=len(train)/len(val)
        """
        # assert len(data_list)!=0 ,'Null data list'
        # assert len(label_list)!=0 ,'Null label list'
        self.data=data_list
        
        self.label=label_list
    
    
    
        transformslist=[
        transforms.Compose([
        transforms.RandomCrop(512, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])]
    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return super().__len__()
        
    # @staticmethod
    # def defaultloader(path):
    #     image= Image.open(path)
    #     image=image.convert('RGB')
    #     tensor

























def namefilter(self, name, key):
    if name.find(key) == -1:
        return False
    else:
        return True


class dataset_generator():
    def __init__(self,root,filtypes=['png','jpg','jepg']):
        print()
        super(dataset_generator,self).__init__()
        print('******* dataset builder init info ******')
        assert os.path.exists(root),'Invalid Root Path'
        assert os.path.exists(os.path.join(root,'image')),'cant find image folder'
        assert os.path.exists(os.path.join(root,'label')),'cant find label folder'
        self.root=root
        self.imagefolder=os.path.join(root,'image')
        self.labelfolder=os.path.join(root,'label')
        print('\nimagedir:',self.imagefolder,'\nlabel dir:',self.labelfolder)
        self.filetypes=filtypes


    def TraversalFileInPath(self,Key=None,topdown=False):

        """
        traversal all file with specified filetype
        topdown :=True traversal subfolder or not
        Key :=filter with keyword
        """
        objlist=self.filetypes
        dirname=self.imagefolder
        print('************traversal dir ', dirname, '***************\n\nfiletypes:',self.filetypes)
        postfix = set(objlist)
        dirlist = []
        if topdown:
            for maindir, subdir, file_name_list in os.walk(dirname):

                for filename in file_name_list:

                    apath = os.path.join(maindir, filename)
                    # print(apath)
                    if apath.split('.')[-1] in postfix:
                        # print('Avaliable Path : '+apath+' ||')
                        if Key != None:
                            if namefilter(apath, Key):
                                dirlist.append(apath)
                        else:
                            dirlist.append(apath)
        if not topdown:
            for file in os.listdir(dirname):
                file_path = os.path.join(dirname, file)

                if file_path.split('.')[-1] in postfix:
                    if Key != None:
                        if namefilter(file_path, Key):
                            dirlist.append(file_path)
                    else:
                        dirlist.append(file_path)

        self.dirlist=dirlist
        print ('------imageset process done & info :\n\n')
        print ('image count : ',len(dirlist))


        return dirlist

    def Indexlabel(self):
        """
        image label pair generator

        :return:
        """
        indexpair=[]
        pixelvalues=[]
        for path in self.dirlist:
            indexlabelname = os.path.join(self.labelfolder,path.split('.')[-2].split('/')[-1])
            count=0
            for type in self.filetypes:
                if os.path.exists(indexlabelname+'.'+type):
                    pair={}
                    pair['image']=path
                    labelpath=indexlabelname+'.'+type
                    pair['label']=labelpath
                    image=cv2.imread(labelpath)
                    Max=image[:,:].max()

                    print(image[:,:].max())

                    indexpair.append(pair)
        self.indexpair=indexpair
        print('image & label dict list: ',indexpair)
        return indexpair


    def loader():
        pass






def main():
    print('\n\n------Basic DataSet Builder-----\n\n')
    
    parser = argparse.ArgumentParser(description=
    "------PyTorch Dataset Building Tool Help Notes------")
 
    parser.add_argument('--root',type=str,default='./dataset',help='must input image dir (default is ./dataset )')

    parser.add_argument('--labeldir',type=str,default='./dataset/label',help='must input label dir (default is ./dataset/label)')

    parser.add_argument('--ratio',type=float,default=0.8,help='ratio of train data & val data (default=0.8)')

    # parser.add_argument('--filetype',type=str,default='png,jpg,jepg',help='dataset image file types')
    # parser.add_argument('--')

    args = parser.parse_args()
    # filetypes=args.filetype.split(',')
    print('args input root dir is  :',args.root)
    """
    Process flow:
    1.Traversal image folder & index image file & label file
     
    """
    # dataset=datasetbuilder(args.root)
    # dataset.TraversalFileInPath()
    # dataset.Indexlabel()
    cds=custom_dataset()

if __name__ == '__main__':
    main()
    