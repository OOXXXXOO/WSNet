import glob
import os
import sys
import numpy as np
import pandas as pd
import tifffile as tif
import matplotlib.pyplot as plt
from tqdm import tqdm
localpath="/workspace/data/38cloud-cloud-segmentation-in-satellite-images/38-Cloud_training/"
localpathR="/workspace/data/38cloud-cloud-segmentation-in-satellite-images/38-Cloud_training/train_red/" 
localpathG="/workspace/data/38cloud-cloud-segmentation-in-satellite-images/38-Cloud_training/train_green/"
localpathB="/workspace/data/38cloud-cloud-segmentation-in-satellite-images/38-Cloud_training/train_blue/"

localpathGT="/workspace/data/38cloud-cloud-segmentation-in-satellite-images/38-Cloud_training/train_gt/"

patchs=pd.read_csv(os.path.join(localpath,"training_patches_38-Cloud.csv"))

print(patchs.iloc[:8,:])

def fast_percentager_strentching(image,percentage=1,sample=10000):
    """
    Image ndarray:(W,H,C)
    Percentage N(0-100)%
    """
    assert not percentage>100 or percentage<0,"Invalde Percentage Value"
    # print("-------------------------- percentager_strentching -------------------------")
    # print("----- process with percentage : ",percentage,"% -----")
    percentage=percentage/100
    # print()
    W,H=image.shape[0],image.shape[1]
    w=np.random.randint(0,W,sample)
    h=np.random.randint(0,H,sample)
    if len(image.shape)==3:
        points=image[w,h,:]
        point=[np.mean(channels) for channels in points]
    else:
        points=image[w,h]
        point=points
    pointset=sorted(point)
    min=int(sample*percentage)
    max=int(sample*(1-percentage))
    min=pointset[min]
    max=pointset[max]
    image[image>max]=max
    image[image<min]=min
    image=(image-min)/(max-min)
    # print("----- Max : ",max," Min :    ",min,"-----")
    return image
blue="blue_"
red="red_"
green="green_"
gt="gt_"
whole_patch=[]
index=0
for patch in tqdm(sorted(patchs["name"].iteritems(),key=lambda x: int(x[1].split('_')[1]))):
    Rfilename=localpathR+red+patch[1]+".TIF"
    Gfilename=localpathG+green+patch[1]+".TIF"
    Bfilename=localpathB+blue+patch[1]+".TIF"
    GTfilename=localpathGT+gt+patch[1]+".TIF"
    R=fast_percentager_strentching(tif.imread(Rfilename))
    G=fast_percentager_strentching(tif.imread(Gfilename))
    B=fast_percentager_strentching(tif.imread(Bfilename))
    GT=tif.imread(GTfilename)
    rgb=np.stack([R,G,B],2)
    if R.sum()>0:
        data={
            "imagery":fast_percentager_strentching(rgb),
            "target":GT
        }
        whole_patch.append(data)
        index+=1
    if len(whole_patch)%1000==0:
        print("Saving...")
        np.save("/workspace/data/38cloud-cloud-segmentation-in-satellite-images/cloud1percent"+str(index)+'.npy',whole_patch,allow_pickle=True)
        whole_patch=[]










    #     figure,ax=plt.subplots(1,2,sharex=True,sharey=True)
    #     ax[0].imshow(GT)
    #     ax[1].imshow(fast_percentager_strentching(rgb))
    #     plt.show()