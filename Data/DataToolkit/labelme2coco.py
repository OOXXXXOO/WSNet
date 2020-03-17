# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    labelme2coco.py                                    :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/28 11:47:12 by winshare          #+#    #+#              #
#    Updated: 2020/02/28 11:50:17 by winshare         ###   ########.fr        #
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
class labelme2coco(object):
    def __init__(self,labelme_json=[],outputpath='./',visualization=False):
        '''
        :param labelme_json: 所有labelme的json文件路径组成的列表
        :param save_json_path: json保存位置
        '''
        self.labelme_json=labelme_json
        self.save_json_path=outputpath
        self.images=[]
        self.categories=[]
        self.annotations=[]
        self.label=['__background__']
        self.annID=1
        self.height=0
        self.width=0
        self.visualization=visualization

        self.save_json()
        with open(outputpath+'labels.txt',"w") as labelfp:
            self.label=[i+"\n" for i in self.label]
            labelfp.writelines(self.label)
            labelfp.close()

    def data_transfer(self):
        label_name_to_value = {'_background_': 0}
        for num,json_file in enumerate(self.labelme_json):
            print('processing file :',json_file)
            lbl=None
            with open(json_file,'r') as fp:
                data = json.load(fp)  # 加载json文件

                image={}
                img = utils.image.img_b64_to_arr(data['imageData'])  # 解析原图片数据
                height, width = img.shape[:2]
                
                image['height']=height
                image['width'] = width
                image['id']=num+1
                image['file_name'] = data['imagePath'].split('/')[-1]

                self.height=height
                self.width=width


                self.images.append(image)
                num=0
                for shape in tqdm(sorted(data['shapes'], key=lambda x: len(x['points']),reverse=True)):
                    
                    label_name = None
                    label_name = shape['label'] 
                    if label_name in label_name_to_value.keys():
                        label_value = label_name_to_value[label_name]
                    else:
                        self.categories.append(self.categorie(label_name))
                        self.label.append(label_name)
                        label_value = len(label_name_to_value)
                        label_name_to_value[label_name] = label_value
                    
                    points=shape['points']
                    self.annotations.append(self.annotation(points,label_name,num))
                    self.annID+=1
                    num+=1

                lbl, _ = utils.shapes_to_label(
                    img.shape, data['shapes'], label_name_to_value
                )

                labelpath=os.path.join(self.save_json_path,"./label/")
                if not os.path.exists(labelpath):
                    print("Create path in ",labelpath)
                    os.makedirs(labelpath)
            
                filename=image['file_name'].split('.')[0]
                
                labelfile=labelpath+filename+"_label.png"
                print("write in:",labelfile)
                cv2.imwrite(labelfile,lbl)
                if self.visualization:
                    vizpath=os.path.join(self.save_json_path,"./viz/")
                    if not os.path.exists(vizpath):
                        os.makedirs(vizpath)
                    lbl_viz = imgviz.label2rgb(
                        label=lbl,
                        img=imgviz.rgb2gray(img),
                        label_names=self.label,
                        font_size=30,
                        loc='rb',
                    )
                    plt.imshow(lbl_viz),plt.show()
                    plt.imsave(vizpath+filename+"_viz_label.png",lbl_viz)
                    

                

    def categorie(self,label):
        categorie={}
        categorie['supercategory'] = label[0]
        categorie['id']=len(self.label)+1 # 0 默认为背景
        categorie['name'] = label[1]
        return categorie

    def annotation(self,points,label,num):
        annotation={}
        annotation['segmentation']=[list(np.asarray(points).flatten())]
        annotation['iscrowd'] = 0
        annotation['image_id'] = num+1
        # annotation['bbox'] = str(self.getbbox(points)) # 使用list保存json文件时报错（不知道为什么）
        # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] 使用该方式转成list
        annotation['bbox'] = list(map(float,self.getbbox(points)))

        annotation['category_id'] = self.getcatid(label)
        annotation['id'] = self.annID
        return annotation

    def getcatid(self,label):
        for categorie in self.categories:
            if label[1]==categorie['name']:
                return categorie['id']
        return -1

    def getbbox(self,points):
        # img = np.zeros([self.height,self.width],np.uint8)
        # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA)  # 画边界线
        # cv2.fillPoly(img, [np.asarray(points)], 1)  # 画多边形 内部像素值为1
        polygons = points
        mask = self.polygons_to_mask([self.height,self.width], polygons)
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
        # 解析左上角行列号
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x

        # 解析右下角行列号
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)

        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
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
    parsers.add_argument("--o",default="annotation.json",type=str, help="dir of output annotation file like annotation.json")
    parsers.add_argument("--v",default=False,type=bool, help="bool type about output label visualization or not")
    args = parsers.parse_args()
    return args




def main():
    args=parser()
    anno=args.a
    output=args.o
    labelme_json=glob.glob(anno+'*.json')
    labelme2coco(labelme_json=labelme_json,outputpath=output,visualization=args.v)

if __name__ == '__main__':
    main()
    
    