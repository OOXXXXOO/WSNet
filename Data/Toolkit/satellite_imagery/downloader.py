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

# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    superdownloader.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/05/25 15:10:08 by winshare          #+#    #+#              #
#    Updated: 2020/05/25 15:10:08 by winshare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# ---------------------------------- STD Lib --------------------------------- #

from threading import Thread
import cv2
import multiprocessing
import time
import os
import io
import PIL.Image as Image
import numpy as np
from tqdm import tqdm

# -------------------------------- Custom Lib -------------------------------- #
import urllib.request as ur
from subscription import MAP_URLS,Tilesize
from transform import getExtent,wgs_to_tile,TileXYToQuadKey,saveTiff,wgs_to_mercator






class downloader(Thread):
    def __init__(self, server=None,thread_count=4,format='tif'):
        print("# ---------------------------------------------------------------------------- #")
        print("#                            MAP Production Toolkit                            #")
        print("# ---------------------------------------------------------------------------- #")
        # print("Support DataSources",MAP_URLS.keys())
        self.thread_count=thread_count
        self.format=format
        if server is not None and server in MAP_URLS.keys():

            print(
                "# ---------------------- MAP Serverv Init Successful by ---------------------- #")
            lines = 52
            length = len(server)
            space = lines - length
            print("# ----------------------", server, space * "-", "#")
            self.server = server
        else:
            self.server = "Google"



    def add_cord(self, left, top, right, bottom, zoom, style='s'):
        """
        Compute the Extent by (x1,y1,x2,y2)-Retangle coordinate A
        I: coordinate A->WGS Extent of tile
        II: Extent->Tile coordinate 
        III:Tile coordinate-> array of x,y coordinates WGS left top point in each tile
        """

        # ------------------------------------- I ------------------------------------ #

        self.left, self.top, self.right, self.bottom, self.zoom = left, top, right, bottom, zoom
        self.extent = getExtent(
            self.left,
            self.top,
            self.right,
            self.bottom,
            self.zoom,
            self.server)

        # ------------------------------------ II ------------------------------------ #

        self.pos1x,self.pos1y = wgs_to_tile(self.left, self.top, self.zoom)
        pos2x, pos2y = wgs_to_tile(self.right, self.bottom, self.zoom)
        self.lenx = pos2x - self.pos1x + 1
        self.leny = pos2y - self.pos1y + 1
        width = self.lenx * Tilesize
        height = self.leny * Tilesize
        left_top_x,left_top_y=self.extent["LT"][0],self.extent["LT"][1]
        self.resolution_x=(self.extent['RB'][0] - self.extent['LT'][0]) / width
        self.resolution_y=(self.extent['RB'][1] - self.extent['LT'][1]) / height
        right_bottom_x=left_top_x+width*self.resolution_x
        right_bottom_y=left_top_y+height*self.resolution_y
        self.wsg_cord=(left_top_x,left_top_y,right_bottom_x,right_bottom_y)
      
        self.mercator_cord=(*wgs_to_mercator(left_top_x,left_top_y),*wgs_to_mercator(right_bottom_x,right_bottom_y))


        print('# -----WGS BoundingBox:',self.wsg_cord)
        print("# -----Mercator BoudingBox:",self.mercator_cord)
        # ------------------------------------ III ----------------------------------- #

        self.x_step_point=np.arange(left_top_x,right_bottom_x,self.resolution_x*Tilesize)
        self.y_step_point=np.arange(left_top_y,right_bottom_y,self.resolution_y*Tilesize)
      
        # ----------------------------- URL -> URL Queue ----------------------------- #

        self.urls=self.get_urls(left, top, right, bottom, zoom, style)
        self.urls_queue=multiprocessing.Manager().Queue()
        for data in self.urls:
            self.urls_queue.put(data)

        print("# -----Url Queue size:",self.urls_queue.qsize())


    def downloadurl(self,urls_queue):

        # -------------------------- process the single url -------------------------- #
        t_start = time.time()
        instance=urls_queue.get()
        url=instance["url"]
        HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/29.0.1547.76 Safari/537.36'
            }
        header = ur.Request(url, headers=HEADERS)
        err = 0
        while(err < 3):
            try:
                data = ur.urlopen(header).read()
            except BaseException:
                err += 1
            else:
                picio = io.BytesIO(data)
                small_pic = Image.open(picio)
                smallarray = np.array(small_pic)
                
                info=instance["info"]
                x=info[0]
                y=info[1]
                xx=self.x_step_point[x-self.pos1x]
                yy=self.y_step_point[y-self.pos1y]

                Proj=[
                    xx,
                    self.resolution_x,
                    0,
                    yy,
                    0,
                    self.resolution_y
                ]
            
                name="{server}-{x}-{y}-{z}".format(x=info[0],y=info[1],z=info[2],server=self.server)
                path=os.path.join(self.output_path,name)

                # ------------------------------------ I/O ----------------------------------- #

                if self.format=='png':
                    path+='.png'
                    cv2.imwrite(path,smallarray)
                if self.format=='tif':
                    path+='.tif'
                    if len(smallarray.shape) == 3:
                        h, w, c = smallarray.shape
                        if c == 3:
                            r, g, b = cv2.split(smallarray)
                            saveTiff(r, g, b, Proj, path)
                        if c == 4:
                            r, g, b, a = cv2.split(smallarray)
                            saveTiff(r, g, b, Proj, path)
                    if len(smallarray.shape) == 2:
                        saveTiff(
                        smallarray,
                        smallarray,
                        smallarray,
                        Proj,
                        path)
                # print(path,' || save done ')
                instance["path"]=path
                t_stop = time.time() 
                return instance

        raise Exception("Bad network link.")
      

    def download(self,output_path="./images"):   
        # print('# -----Queue Downloading with ',self.thread_count,' Process')
        self.output_path=output_path
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        result_vector=[]


        # -------------------------------- Thread Pool ------------------------------- #

        pool=multiprocessing.Pool(self.thread_count)
        for index,url in enumerate(self.urls):
            result=pool.apply_async(self.downloadurl,args=(self.urls_queue,))
            result_vector.append(result) 
            if index%100==0:
                time.sleep(5)
                print("# ===== download (",index ,") to :",result)


        pool.close()
        pool.join()

        # ------------------------------ process result ------------------------------ #

        self.result=[i.get() for i in result_vector]        
        print("# ------------------------------- Download Done ------------------------------ #")

        name="{server}-{time_}-{rect}-{zoom}.json".format(
            server=self.server,
            time_=str(time.asctime(time.localtime(time.time()))),
            rect=str((self.left,self.top,self.right,self.bottom)),
            zoom=self.zoom
        )
        json_path=os.path.join(output_path,name)
        self.json_path=json_path
        tileinfo={
            "time":str(time.asctime(time.localtime(time.time()))),
            "left":self.left,
            "top":self.top,
            "right":self.right,
            "bottom":self.bottom,
            "zoom":self.zoom,
            "server":self.server,
            "data":self.result
            }

        # ------------------- save the meta information for result ------------------- #

        import json
        with open(json_path,"w") as jfp:
            json.dump(tileinfo,jfp)
            print("# ===== Save description done",json_path)

     
    def get_urls(self,x1, y1, x2, y2, z, style):

        # ------------------------------- Get url list ------------------------------- #

        print("# -----Total tiles number：{x} X {y}".format(x=self.lenx, y=self.leny))
        self.urls = [self.get_url(
                i,
                j,
                z,
                style) for j in range(self.pos1y,self.pos1y +self.leny) for i in range(self.pos1x,self.pos1x +self.lenx)
                ]
        return self.urls

    def get_url(self, x, y, z, style):

        # ------------------------------ Resources Check ----------------------------- #

        if self.server == 'Google China':
            url = MAP_URLS["Google China"].format(x=x, y=y, z=z, style=style)
        elif self.server == "Bing VirtualEarth":
            quadkey = TileXYToQuadKey(x, y, z)
            url = MAP_URLS["Bing VirtualEarth"].format(q=quadkey)
        elif self.server == 'Google':
            url = MAP_URLS["Google"].format(x=x, y=y, z=z, style=style)
        else:
            url = MAP_URLS[self.server].format(x=x, y=y, z=z)

        data={}
        data["server"]=self.server
        data["info"]=[x,y,z]
        data["url"]=url
        return data

    def merge(self,path="./result.tif"):

        # ---------------------- Merge tiles to whole tiff file ---------------------- #

        output_array=Image.new("RGBA",(self.lenx*Tilesize,self.leny*Tilesize))
        assert not (self.result==None or len(self.result)==0),"Invalide result of process please check the result of download "+str(self.result)
        for i in self.result:
            x,y,z=i["info"]
            small=Image.open(i['path'])
            x=(x-self.pos1x)*Tilesize
            y=(y-self.pos1y)*Tilesize
            output_array.paste(small,(x,y))

        output_array=output_array.convert("RGB")
        r, g, b = cv2.split(np.array(output_array))
        gt=(
            self.extent['LT'][0],
            (self.extent['RB'][0] - self.extent['LT'][0]) / r.shape[1],
            0,
            self.extent['LT'][1],
            0,
            (self.extent['RB'][1] - self.extent['LT'][1]) / r.shape[0]
        )
        saveTiff(r,g,b,gt,path)

        print('# ----------------------------- Merge tiles done ----------------------------- #')



def main():
    Google=downloader("Google Satellite")
    Google.add_cord(116.3, 39.9, 116.6, 39.7, 13)#WGS Form
    Google.download()
    Google.merge()
    tiles=[i["path"] for i in Google.result]
    from vector import Vector
    Building=Vector('/home/winshare/Downloads/2017-07-03_asia_china.mbtiles')
    Building.getDefaultLayerbyName("building")  
    Building.crop_default_layer_by_rect(Google.mercator_cord)#FILTER to speed up
    Building.generate(tiles)




if __name__ == '__main__':
    main()
    