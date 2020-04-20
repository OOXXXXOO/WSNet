import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import torch
import matplotlib.pyplot as plt
class Costum_NPY_Instance(Dataset):
    def __init__(self,npy=None,Size=512,DatasetLength=5000):
        self.npy=npy
        self.raw=np.load(npy,allow_pickle=True)
        self.length=len(self.raw)
        print("raw length :",self.length)
        self.Size=512
        self.DatasetLength=DatasetLength
        """
        n*Dict
        """
    def __getitem__(self,index):
        data=self.raw[index%self.length]
        # print(data)
        image=data["image"]
        label=data["label"]

        (H,W,_)=image.shape
        x=np.random.randint(0, W-self.Size)
        y=np.random.randint(0, H-self.Size)
        patch_size=self.Size
        image_=image[x:x+patch_size,y:y+patch_size,:]
        label_=label[x:x+patch_size,y:y+patch_size]*255
        _, contours, hierarchy= cv2.findContours(label_,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        plt.imshow(label_),plt.show()
        boxes=[]
        labels=[]
        segmentation=[]
        for points in contours:
            segmentation.append(points)
            x,y,w,h=cv2.boundingRect(points)
            boxes.append((x,y,x+w,y+h))
            labels.append(1)
        anno={
            "boxes":torch.tensor(boxes,dtype=torch.float),
            "labels":torch.tensor(labels,dtype=torch.int64),
            "masks":torch.tensor(segmentation,dtype=torch.uint8)
        }
        boxes=None
        labels=None
        segmentation=None
        return image_,anno 
    def __len__(self):
        return self.DatasetLength



def main():
    npy=Costum_NPY_Instance("/home/winshare/Experiment/solar.npy")
    print(npy[0])


if __name__ == '__main__':
    main()
    