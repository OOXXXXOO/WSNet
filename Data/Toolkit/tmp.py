# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    tmp.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/28 11:47:12 by winshare          #+#    #+#              #
#    Updated: 2020/04/14 10:50:14 by winshare         ###   ########.fr        #
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



import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from labelme import utils
import numpy as np
import glob
import PIL.Image
from tqdm import tqdm
import imgviz
import os
import tifffile as TIF


# t1=TIF.imread(tif[0])[:5093,:3337,:3]
# t2=TIF.imread(tif[1])[10134:,:5090,:3]
# t3=TIF.imread(tif[2])[4346:,:6149,:3]
# t4=TIF.imread(tif[3])[:2968,:7212,:3]


class labelme2coco(object):
    def __init__(self,labelme_json=[],outputpath='./',visualization=False,mode="XYWH_ABS",only_mask=False):
        '''
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.mode=mode
        self.labelme_json=labelme_json
        self.save_json_path=outputpath
        self.images=[]
        self.categories=[]
        self.annotations=[]
        self.label=['__background__',"Photovoltaic panels"]

        self.annID=0
        self.height=0
        self.width=0
        self.visualization=visualization
        self.only_mask=only_mask
        print("-----visualization :",self.visualization ,"only mask:",self.only_mask)
        self.save_json()
        with open(outputpath+'labels.txt',"w") as labelfp:
            self.label=[i+"\n" for i in self.label]
            labelfp.writelines(self.label)
            labelfp.close()

    def data_transfer(self):
        label_name_to_value = {'_background_': 0,"Photovoltaic panels":1}
        for num,json_file in enumerate(sorted(self.labelme_json)):
            print('processing file :',json_file)
            lbl=None
            with open(json_file,'r') as fp:
                data = json.load(fp)  # 加载json文件
    
                fp.close()

                image={}
                if num==0:

                    img = utils.image.img_b64_to_arr(data['imageData'])  # 解析原图片数据
                    (self.ox,self.oy,_)=img.shape
                    img=img[:5093,:3337,:3]
                    self.sx,self.sy=0,0

                if num==1:
                    
                    img = utils.image.img_b64_to_arr(data['imageData'])  # 解析原图片数据
                    # plt.imshow(img),plt.show()
                    (self.ox,self.oy,_)=img.shape
                    img=img[10134:,:5090,:3]
                    self.sx,self.sy=0,10134
                if num==2:
                    img = utils.image.img_b64_to_arr(data['imageData'])  # 解析原图片数据
                    (self.ox,self.oy,_)=img.shape
                    img=img[4346:,:6149,:3]
                    self.sx,self.sy=0,4346
                if num==3:
                    img = utils.image.img_b64_to_arr(data['imageData'])  # 解析原图片数据
                    (self.ox,self.oy,_)=img.shape
                    img=img[:2968,:7212,:3]
                    self.sx,self.sy=0,0

                print("image shape is ",img.shape)
                # print(img.shape)
                # plt.imshow(img),plt.show()
                height, width = img.shape[:2]
        
                
                image['height']=height
                image['width'] = width
                image['id']=num
                image['file_name'] = data['imagePath'].split('/')[-1]

                self.height=height
                self.width=width
                SHPS=[]
    
                if not self.only_mask:
                    self.images.append(image)
                    # self.show_memory()
                    print("----- Shapes Processing:")
                    
                    for shape in tqdm(sorted(data['shapes'], key=lambda x: len(x['points']),reverse=True)):
                        label_name = None
                        label_name = shape['label'] 
                        # if label_name in label_name_to_value.keys():
                        #     label_value = label_name_to_value[label_name]
                        # else:
                        #     self.categories.append(self.categorie(label_name))
                        #     self.label.append(label_name)
                        #     label_value = len(label_name_to_value)
                        #     label_name_to_value[label_name] = label_value
                        
                        points=np.asarray(shape['points'])
                        # points_=points.copy()
                        # print(points)
                        points[:,0]-=self.sx
                        points[:,1]-=self.sy
                        # print(points)
                        shape["points"]=points
                        SHPS.append(shape)
                        self.annotations.append(self.annotation(points,label_name,num))
                        self.annID+=1
                    # self.show_memory()

                if self.visualization:
                    print("-----ImageLabel Processing:")
                    # shps=data["shapes"]
                    # shps[i[:,0]- for i in shps]
                    lbl, _ = utils.shapes_to_label(
                        (height,width,3),data["shapes"], label_name_to_value
                    )
                    # plt.imshow(lbl),plt.show()


                    labelpath=os.path.join(self.save_json_path,"./label/")
                    if not os.path.exists(labelpath):
                        print("Create path in ",labelpath)
                        os.makedirs(labelpath)
                
                    filename=image['file_name'].split('.')[0]
                    labelfile=os.path.join(labelpath,filename+"_label.png")
                    print("-----write in:",labelfile)
                    print("-----lbl Saving....")
                    cv2.imwrite(labelfile,lbl)
                    print("-----lbl Save Done")
                    

                    vizpath=os.path.join(self.save_json_path,"./viz/")
                    if not os.path.exists(vizpath):
                        os.makedirs(vizpath)
                    lbl_viz = imgviz.label2rgb(
                        label=lbl,
                        img=imgviz.rgb2gray(img)
                    )
                    img=None
                    # plt.imshow(lbl_viz),plt.show()
                    plt.imsave(vizpath+filename+"_viz_label.png",lbl_viz)
                data=None
                    

                

    def categorie(self,label):
        categorie={}
        categorie['supercategory'] = label[0]
        categorie['id']=len(self.label) # 0 默认为背景
        categorie['name'] = label
        return categorie

    def annotation(self,points,label,num):
        annotation={}
        annotation['segmentation']=[list(np.asarray(points).flatten())]
        annotation['iscrowd'] = 0
        annotation['image_id'] = num
        annotation['bbox'] = list(map(float,self.getbbox(points)))
        annotation['category_id'] = self.getcatid(label)
        annotation['id'] = self.annID
        return annotation

    def getcatid(self,label):
        for categorie in self.categories:
            
            if label==categorie['name']:
                # print('====',label,categorie['name'],categorie['id'])
                return categorie['id']

        return -1

    def getbbox(self,points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points
        mask = self.polygons_to_mask([self.height,self.width], polygons)
        # plt.imshow(mask),plt.show() /
        # print(mask.shape)/
        return self.mask2box(mask)

    def mask2box(self, mask):
        '''从mask反算出其边框
        mask：[h,w]  0、1组成的图片
        1对应对象，只需计算1对应的行列号（左上角行列号，右下角行列号，就可以算出其边框）
        '''
        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
        # print(rows)
        # print(clos)
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        if self.mode=="XYXY_ABS":
            return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        if self.mode=="XYWH_ABS":
            return [left_top_c, left_top_r, right_bottom_c-left_top_c, right_bottom_r-left_top_r]  # [x1,y1,w,h] 对应COCO的bbox格式

    def polygons_to_mask(self,img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask

    def data2coco(self):
        data_coco={}
        data_coco['images']=self.images
        data_coco['categories']=self.categories
        data_coco['annotations']=self.annotations
        return data_coco

    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
        # 保存json文件
        print("whole annotation file already save in :",self.save_json_path)
        json.dump(self.data_coco, open(self.save_json_path+"annotation.json", 'w'), indent=4)  # indent=4 更加美观显示



def parser():
    parsers=argparse.ArgumentParser()
    parsers.add_argument("--a",default="./annotation/", type=str,help="dir of anno file like ./annotation/")
    parsers.add_argument("--o",default="./",type=str, help="dir of output annotation file like annotation.json")
    parsers.add_argument("--v",default=False,type=bool, help="bool type about output label visualization or not")
    parsers.add_argument("--m",default="XYWH_ABS",type=str, help="box format mode like support : XYWH_ABS(Default for COCO) , XYXY_ABS")
    parsers.add_argument("--l",default=False,type=bool, help="only output mask")

    args = parsers.parse_args()
    return args




def main():
    args=parser()
    anno=args.a
    output=args.o
    mode=args.m
    visualization=args.v
    only_mask=args.l
    if not os.path.exists(output):
        print("-----Create empty in :",output)
    labelme_json=glob.glob(anno+'*.json')
    print("-----Para:","\nAnnotations:",labelme_json,
    "\n\nOutput dir:",output,
    "\n\nMode :",mode)
    print("\n-----Process :")
    labelme2coco(labelme_json=labelme_json,outputpath=output,visualization=visualization,mode=mode,only_mask=only_mask)

if __name__ == '__main__':
    main()
    
    