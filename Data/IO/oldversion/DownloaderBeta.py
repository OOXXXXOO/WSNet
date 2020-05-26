import numpy as np
import math
import gdal
from math import floor, pi, log, tan, atan, exp

import urllib.request as ur
import PIL.Image as pil
import matplotlib.pyplot as plt
import io
import cv2
import osr
import multiprocessing
import time
from tqdm import tqdm
import os
import sys
# ------------------Interchange between WGS-84 and Web Mercator-----------
from Transform import *



# ---------------------------------------------------------

# ---------------------------------------------------------

class DOWNLOADER():
    def __init__(self, server=None):
        """
        Download images based on spatial extent.
        East longitude is positive and west longitude is negative.
        North latitude is positive, south latitude is negative.
        Parameters
        ----------
        left, top : left-top coordinate, for example (100.361,38.866)

        right, bottom : right-bottom coordinate

        z : zoom
        filePath : File path for storing results, TIFF format

        style :
            m for map;
            s for satellite;
            y for satellite with label;
            t for terrain;
            p for terrain with label;
            h for label;

        source : Google China (default) or Google
        Toolkit Support DataSources:

        [
            'Google'
            'Google China',
            'Google Maps',
            'Google Satellite',
            'Google Terrain',
            'Google Terrain Hybrid',
            'Google Satellite Hybrid'
            'Stamen Terrain'
            'Stamen Toner'
            'Stamen Toner Light'
            'Stamen Watercolor'
            'Wikimedia Map'
            'Wikimedia Hike Bike Map'
            'Esri Boundaries Places'
            'Esri Gray (dark)'
            'Esri Gray (light)'
            'Esri National Geographic'
            'Esri Ocean',
            'Esri Satellite',
            'Esri Standard',
            'Esri Terrain',
            'Esri Transportation',
            'Esri Topo World',
            'OpenStreetMap Standard',
            'OpenStreetMap H.O.T.',
            'OpenStreetMap Monochrome',
            'OpenTopoMap',
            'Strava All',
            'Strava Run',
            'Open Weather Map Temperature',
            'Open Weather Map Clouds',
            'Open Weather Map Wind Speed',
            'CartoDb Dark Matter',
            'CartoDb Positron',
            'Bing VirtualEarth'
        ]

        """

        print("# ---------------------------------------------------------------------------- #")
        print("#                            MAP Production Toolkit                            #")
        print("# ---------------------------------------------------------------------------- #")
        # print("Support DataSources",MAP_URLS.keys())
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

    # def addcord(self, left, top, right, bottom, zoom, style='s'):
    #     self.left, self.top, self.right, self.bottom, self.zoom = left, top, right, bottom, zoom
    #     self.extent = getExtent(
    #         self.left,
    #         self.top,
    #         self.right,
    #         self.bottom,
    #         self.zoom,
    #         self.server)
    #     # Get the urls of all tiles in the extent
    #     urls = get_urls(left, top, right, bottom, zoom, self.server, style)
    #     # Group URLs based on the number of CPU cores to achieve roughly equal
    #     # amounts of tasks
    #     self.urls_group = [urls[i:i + math.ceil(len(urls) / multiprocessing.cpu_count())]
    #                        for i in range(0, len(urls), math.ceil(len(urls) / multiprocessing.cpu_count()))]

    #     return self.urls_group


    def add_cord(self, left, top, right, bottom, zoom, style='s'):
        self.left, self.top, self.right, self.bottom, self.zoom = left, top, right, bottom, zoom
        self.extent = getExtent(
            self.left,
            self.top,
            self.right,
            self.bottom,
            self.zoom,
            self.server)
        # Get the urls of all tiles in the extent
        self.urls = get_urls(left, top, right, bottom, zoom, self.server, style)
        import queue
        self.urls_queue=queue.Queue()
        for url in self.urls:
            self.urls_queue.put(url)
        print('-----Length of Url Queue:',self.urls_queue.qsize())



    def download(self):
        # Each set of URLs corresponds to a process for downloading tile maps
        print('Tiles downloading......')
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        results = pool.map(download_tiles, self.urls)
        pool.close()
        pool.join()
        # self.result = [x for j in results for x in j]
        print(results)
        self,results=results
        print('Tiles download complete')
        return self.result

    # Combine downloaded tile maps into one map
    def merge(self, dir='./', filename=None):

        print("# ---------------------------------------------------------------------------- #")
        print("#                                     Merge                                    #")
        print("# ---------------------------------------------------------------------------- #")

        outpic = merge_tiles(
            self.result,
            self.left,
            self.top,
            self.right,
            self.bottom,
            self.zoom)
        outpic = outpic.convert('RGB')
        r, g, b = cv2.split(np.array(outpic))
        # Get the spatial information of the four corners of the merged map and
        # use it for outputting
        self.extent = getExtent(
            self.left,
            self.top,
            self.right,
            self.bottom,
            self.zoom,
            self.server)
        self.gt = (
            self.extent['LT'][0],
            (self.extent['RB'][0] - self.extent['LT'][0]) / r.shape[1],
            0,
            self.extent['LT'][1],
            0,
            (self.extent['RB'][1] - self.extent['LT'][1]) / r.shape[0]
        )

        if filename is not None:
            saveTiff(r, g, b, self.gt, filename)
            print("Save Done", eventpath)
            return filename
        else:
            filename = self.server + str(self.gt) + '.tif'
            eventpath = os.path.join(dir, filename)
            saveTiff(r, g, b, self.gt, eventpath)
            print("Save Done", eventpath)
            return eventpath

        # GT(0)和GT(3)是第一组，表示图像左上角的地理坐标
        # GT(1)和GT(5)是第二组，表示图像横向和纵向的分辨率（一般这两者的值相等，符号相反，横向分辨率为正数，纵向分辨率为负数）；
        # GT(2)和GT(4)是第三组，表示图像旋转系数，对于一般图像来说，这两个值都为0。

    def savetiles(self, path='./', format='tif'):
        if not os.path.exists(path):
            os.makedirs(path)
        pos1x, pos1y = wgs_to_tile(self.left, self.top, self.zoom)
        pos2x, pos2y = wgs_to_tile(self.right, self.bottom, self.zoom)
        lenx = pos2x - pos1x + 1
        leny = pos2y - pos1y + 1
        width = lenx * Tilesize
        height = leny * Tilesize

        self.gt = (
            self.extent['LT'][0],
            (self.extent['RB'][0] - self.extent['LT'][0]) / width,
            0,
            self.extent['LT'][1],
            0,
            (self.extent['RB'][1] - self.extent['LT'][1]) / height
        )

        self.gts = []
        ltx = self.gt[0]
        lty = self.gt[3]
        stepx = self.gt[1]
        stepy = self.gt[5]
        rbx = self.gt[0] + width * stepx
        rby = self.gt[3] + height * stepy
        filelist = []
        
        tilex=np.arange(pos1x,pos2x+1,1)
        tiley=np.arange(pos1y,pos2y+1,1)
        tilelist=[(i,j)  for j in tilex for i in tiley]
        print("TileSize:",len(tilelist))


        x = np.arange(ltx, rbx, step=stepx * Tilesize)
        y = np.arange(lty, rby, step=stepy * Tilesize)

        print("Size:", len(x), "X", len(y))
        
        for h in y:
            for w in x:
                sgt = (w, stepx, 0, h, 0, stepy)
                self.gts.append(sgt)

        if format == 'png':
            for i, data in enumerate(self.result):
                picio = io.BytesIO(data)
                small_pic = pil.open(picio)
                if format == 'png':
                    file = str(i) + ".png"
                    small_pic.save(file)

        if format == 'tif':
            index = 0
            for data in tqdm(self.result):
                picio = io.BytesIO(data)
                small_pic = pil.open(picio)
                small_pic.convert("RGB")
                file = self.server+"-{x}-{y}-{z}" + ".tif"
                file=file.format(x=tilelist[index][0],y=tilelist[index][1],z=self.zoom)
                file = os.path.join(path, file)
                filelist.append(file)
                smallarray = np.array(small_pic)
                # print("shape: ",smallarray.shape)
                if len(smallarray.shape) == 3:
                    h, w, c = smallarray.shape
                    if c == 3:
                        r, g, b = cv2.split(smallarray)
                        saveTiff(r, g, b, self.gts[index], file)
                    if c == 4:
                        r, g, b, a = cv2.split(smallarray)
                        saveTiff(r, g, b, self.gts[index], file)
                if len(smallarray.shape) == 2:
                    saveTiff(
                        smallarray,
                        smallarray,
                        smallarray,
                        self.gts[index],
                        file)

                index += 1

        return filelist

# 改动计划:
# 1,将单线程内处理的任务直接写入文件避免内存爆炸
# 2,将多线程任务分发改为队列分发
# 3,变量一次性定义





def main():

    Google = DOWNLOADER("Google Satellite")
    Google.add_cord(116.3, 39.9, 116.6, 39.7, 13)
    Google.download()
    # tiles = Google.savetiles(path="./Image", format="tif")
    # from Vector import Vector
    # Beijing=Vector("/workspace/data/Water/Beijing.geojson")
    # Beijing.getDefaultLayerbyName("Beijing")
    # Beijing.generate(tiles,output_path='./Label')


if __name__ == '__main__':
    main()
