import tifffile as tif

import numpy as np

import pandas as pd

import glob

import os

import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm


Size=768
Overlap=0.5
sample_ratio=0.05
patch_size=Size
patch=[]
debug=False
def binary_patch(dataset):
    print("-----start generate binary patch")
    devidesub=[]
    Index=0
    for data in dataset:
        print('-----process data : ',Index)
        imagery=data["data"]
        label=data["label"]
        sample_instance(imagery,label)
        Index+=1

def sample(imagery,label):
    print("\n\n-----Patch Size: ",len(patch),"-----\n\n")
    H=imagery.shape[0]-Size
    W=imagery.shape[1]-Size
    num=int((H*W)/(Size**2*sample_ratio))
    print('Data Length is ,',num)
    for i in tqdm(range(num)):
        x=np.random.randint(0, H)
        y=np.random.randint(0, W)
        image_=imagery[x:x+patch_size,y:y+patch_size,:]
        label_=label[x:x+patch_size,y:y+patch_size]

        # label_[label_==4]=1
        # label_[label_>1]=0
        if debug:
            plt.imshow(image_),plt.show()
            plt.imshow(label_),plt.show()

        patch.append(
            {
                "data":image_,
                "label":label_
            }
        )
        
def sample_instance(imagery,label):
    print("\n\n-----Patch Size: ",len(patch),"-----\n\n")
    H=imagery.shape[0]-Size
    W=imagery.shape[1]-Size
    num=int((H*W)/(Size**2*sample_ratio))
    print('Data Length is ,',num)
    dataset=[]
    
    for i in tqdm(range(num)):
        x=np.random.randint(0, H)
        y=np.random.randint(0, W)
        image_=imagery[x:x+patch_size,y:y+patch_size,:]
        label_=label[x:x+patch_size,y:y+patch_size]
        file_name
        sem_seg_file_name
        height, width
        annotations
        bbox
        bbox_mode=BoxMode.XYXY_ABS
        category_id
        segmentation
        image, contours, hierarchy= cv2.findContours(label1*255,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        for points in contours:
            labeldict["segmentation"].append(points)
            x,y,w,h=cv2.boundingRect(points)
            labeldict["boxes"].append((x,y,x+w,y+h))
            labeldict["labels"].append(label_[x,y])
            draw_1=cv2.rectangle(image_, (x,y), (x+w,y+h), 127, 2)
        # print(labeldict)
        dataset.append(data)
        plt.imshow(draw_1),plt.show()


    


def multi_class_patch():
    print("-----start generate multi-class patch")


def generate_basic_patchs(input_=None,label=None):
    """
    The Most important :

    the any path couldn't have any file that not imagery


    
    The name will be sorted ,so the name-rule of files must could be sort and index to same group files
     like 
        input 
            1.tif
            2.tif

        label
            1_label.tif
            2_label.tif
    """
    input_='/workspace/data/Dataset/input/'
    # path of input imagery files like 1.tif 
    label='/workspace/data/Dataset/label/'
    # path of annotation files like 1_label.tif
    
    inputs=sorted(glob.glob(os.path.join(input_,"*.*")))
    labels=sorted(glob.glob(os.path.join(label,"*.*")))
    print('\n\n-----inputs:',inputs)
    print('\n\n-----labels:',labels)

    datasetnp=[]
    for i in tqdm(range(len(inputs))):
        img=cv2.imread(inputs[i])
        lbl=cv2.imread(labels[i])[:,:,0]
        data={
            "data":img,
            "label":lbl
        }
        datasetnp.append(data)
    if len(datasetnp)>0:
        np.save("dataset",datasetnp,allow_pickle=True)



def main():
    """
        data={
            "data":img,
            "label":lbl
        }
    """
    dataset="/workspace/RawDataset.npy"
    dataset_raw=np.load(dataset,allow_pickle=True)[:2]
    binary_patch(dataset_raw)

    # np.save("/workspace/SampledDatasetHTSJ",patch)
    # print("/workspace/SampledDatasetMini.npy is already save")
    


if __name__ == '__main__':
    main()
    