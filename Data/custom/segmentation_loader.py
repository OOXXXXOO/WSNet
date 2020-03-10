import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2


class Costum_NPY_DataSet(Dataset):
    def __init__(self,npy=None,data_ratio=0.8,forward=True,shuffle=True,transforms=None):
        """
        dataset for npy fast - format
        """

        print("-------------------------- Costum_NPY_DataSet Init -------------------------")

        self.npy=npy
        print("\n-----Read: ",len(npy))

        self.length=int(len(self.npy)*data_ratio)
        self.T=transforms
        
        if forward:
            self.npy_=self.npy[:self.length]
        else:
            self.npy_=self.npy[self.length:]
        print("\n-----Read Done, Info : ",len(self.npy_))
        # if shuffle:
        #     np.random.shuffle(self.npy)
        

        
    def __getitem__(self,index):
        image,target=self.npy_[index]["data"],self.npy_[index]["label"]
        image = cv2.resize(image,(512,512), interpolation = cv2.INTER_AREA)
        target= cv2.resize(target,(512,512), interpolation = cv2.INTER_AREA)
        image,target=self.T(image,target)


        return image,target

    def __len__(self):
        return self.length

