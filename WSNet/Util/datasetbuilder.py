import cv2
import torch.nn as nn
import os
import argparse
from Instance import *
import numpy
import torchvision.transforms as transforms


def namefilter(self, name, key):
    if name.find(key) == -1:
        return False
    else:
        return True

class datasetbuilder(Instance):
    def __init__(self,root,filtypes=['png','jpg','jepg']):
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
    dataset=datasetbuilder(args.root)
    dataset.TraversalFileInPath()
    dataset.Indexlabel()


if __name__ == '__main__':
    main()
    