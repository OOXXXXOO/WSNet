# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    temdet.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/04/21 15:51:07 by winshare          #+#    #+#              #
#    Updated: 2020/04/27 17:46:24 by winshare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# Copyright 2020 winshare
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




from skimage import morphology 
import cv2
import glob
import numpy as np
from skimage.color import rgb2gray
from tqdm import tqdm
import time
from .TIF import TIF
class nmsdet():
    def __init__(self,imagery,CHANNEL_INDEX=[0,1,2,3]):
        """
        Input Imagery :
        [C*H*W]
        [H*W*C]
        """
        print("# ---------------------------------------------------------------------------- #")
        print("#                    NMS&Template MApping Toolkit                              #")
        print("# ---------------------------------------------------------------------------- #")
        self.localtime = str(time.asctime(time.localtime(time.time())))
        print('-----Template NMS detect in : \n',imagery,"\n",self.localtime)
        self.imagery=imagery
        self.boxes=[]
        self.templates=[]
        self.ndvi=False
        self.ndvibuilded=False
        imagerytype=self.imagery.split('.')[-1]

        if imagerytype=="tif":
            tif=TIF(filename=self.imagery,channel=CHANNEL_INDEX)
            # tif.fast_percentager_strentching(tif.image)
            self.imagery=tif.image
        elif imagerytype in ["png","jpg","jepg"]:
            self.imagery=cv2.imread(self.imagery)
        else:
            print(self.imagery,"invalid image type : ",imagerytype)
            exit(0)
        print("image shape",self.imagery.shape)
        channels=self.imagery.shape[-1]
        if channels>3:
            self.ndvi=True
            self.Red=self.imagery[:,:,CHANNEL_INDEX[0]]
            self.NIR=self.imagery[:,:,CHANNEL_INDEX[-1]]
        self.RGB=self.imagery[:,:,CHANNEL_INDEX[:3]]


    def addbox(self,x1,y1,x2,y2):
        self.boxes.append((x1,y1,x2,y2))
        if self.ndvibuilded:
            self.templates.append(self.NDVI[y1:y2,x1:x2])
        else:
            self.templates.append(self.imagery[y1:y2,x1:x2,:3])
    
    def nms(self, threshold=0.1):
        bounding_boxes=self.boxes
        confidence_score=self.conf
        # If no bounding boxes, return empty list
        if len(bounding_boxes) == 0:
            return [], []

        # Bounding boxes
        boxes = np.array(bounding_boxes)

        # coordinates of bounding boxes
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]

        # Confidence scores of bounding boxes
        score = np.array(confidence_score)

        # Picked bounding boxes
        picked_boxes = []
        picked_score = []

        # Compute areas of bounding boxes
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)

        # Sort by confidence score of bounding boxes
        order = np.argsort(score)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score
            picked_boxes.append(bounding_boxes[index])
            picked_score.append(confidence_score[index])
            a=start_x[index]
            b=order[:-1]
            c=start_x[order[:-1]]
            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # Compute areas of intersection-over-union
            w = np.maximum(0.0, x2 - x1 + 1)
            h = np.maximum(0.0, y2 - y1 + 1)
            intersection = w * h

            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

            left = np.where(ratio < threshold)
            order = order[left]
        return picked_boxes, picked_score
    def buildndvi(self,NDVIINDEX=1):
        assert self.ndvi,"Invalid Channel for NDVI"
        self.ndvibuilded=True
        self.NDVI=(((self.NIR-self.Red)*NDVIINDEX)/((self.NIR+self.Red)*NDVIINDEX))
        return self.NDVI


    def fast_percentager_strentching(self,image='',percentage=5,sample=10000):
        """
        Image ndarray:(W,H,C)
        Percentage N(0-100)%
        """
        if isinstance(image,str):
            image=self.RGB
        assert not percentage>100 or percentage<0,"Invalde Percentage Value"
        print("-------------------------- percentager_strentching -------------------------")
        print("----- process with percentage : ",percentage,"% -----")
        percentage=percentage/100
        print()
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
        print("----- Max : ",max," Min :    ",min,"-----")
        self.RGB=image
        return image


    def detect(self,threshold = 0.95,drawbox=False,font = cv2.FONT_HERSHEY_SIMPLEX,font_scale = 1,thickness = 2):
        index=0
        img =self.fast_percentager_strentching(image=self.RGB,percentage=1)
 
        for template in tqdm(self.templates):
            h,w=template.shape[:2]
            # plt.imshow(template),plt.show()
            if self.ndvibuilded:
                res = cv2.matchTemplate(self.NDVI.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            else:
                res = cv2.matchTemplate(img.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):  # Optional para
                br = (pt[0] + w, pt[1] + h)
                self.boxes.append((pt[0],pt[1],br[0],br[1]))
            self.conf=np.random.uniform(0.7,0.8,len(self.boxes))
            print("feature ", index," mapping :",len(self.boxes))
            self.boxes,self.score=self.nms(threshold=0.1)

        # ---------------------------------------------------------------------------- #
        #                                 DrawFunction                                 #
        # ---------------------------------------------------------------------------- #

        if drawbox:
            image=self.RGB.copy()
            H,W,_=image.shape
            for (start_x, start_y, end_x, end_y), confidence in tqdm(zip(self.boxes,self.score)):
                cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 255, 255), thickness)
                # (w, h), baseline = cv2.getTextSize(str(confidence), font, font_scale, thickness)
                # cv2.rectangle(org, (start_x, start_y - (2 * baseline + 5)), (start_x + w, start_y), (0, 255, 255), -1)
                
            image=cv2.putText(image, "BoxNMS:"+str(int(len(self.boxes)*1.5)), (int(W*0.8),int(H*0.1)), font, font_scale, (255, 0, 255), thickness)
            # plt.imshow(image),plt.show()
            cv2.imwrite("result"+str(int(len(self.boxes)*1.5))+".png",image)
        return self.boxes,self.score


        
def main():
    detector=nmsdet("/workspace/data/clip/clip1.tif",CHANNEL_INDEX=[0,1,2])
    # ndvi=detector.buildndvi()
    # plt.imshow(ndvi),plt.show()
    detector.addbox(0,0,100,100)
    detector.detect(drawbox=True)
    print("boxes",detector.boxes,"conf",detector.conf)
    detector.addbox(100,100,200,200)
    detector.detect(drawbox=True)
    print("boxes",detector.boxes,"conf",detector.conf)



if __name__ == '__main__':
    main()
    