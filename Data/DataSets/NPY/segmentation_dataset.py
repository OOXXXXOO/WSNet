import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import torch

class Costum_NPY_DataSet(Dataset):
    def __init__(self,npy=None,data_ratio=0.8,forward=True,shuffle=True,transforms=None):
        """
        dataset for npy fast - format
        """

        print("# -------------------------- Costum_NPY_DataSet Init ------------------------- #")

        self.npy=npy
        print("\n-----Read: ",len(npy))

        self.length=int(len(self.npy)*data_ratio)
        self.T=transforms
        
        if forward:
            self.npy_=self.npy[:self.length]
        else:
            self.npy_=self.npy[self.length:]
        self.length=len(self.npy_)
        print("\n-----Read Done, Info : ",len(self.npy_))

        # if shuffle:
        #     np.random.shuffle(self.npy)
        

        
    def __getitem__(self,index):
        image,target=self.npy_[index]["data"],self.npy_[index]["label"]
        target=np.uint8(target/255)
        image=np.uint8(image*255)
        if self.T!=None:
            image=self.T(image).to(torch.float32)
            target=torch.from_numpy(target).to(torch.float32)


        return image,target

    def __len__(self):
        return self.length-1
