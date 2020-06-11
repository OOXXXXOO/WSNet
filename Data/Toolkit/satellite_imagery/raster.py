# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    raster.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: tanwenxuan <tanwenxuan@student.42.fr>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/04/20 17:05:30 by winshare          #+#    #+#              #
#    Updated: 2020/06/02 12:45:00 by tanwenxuan       ###   ########.fr        #
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


import sys
import os
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import gdal
import glob
import matplotlib.pyplot as plt
import numpy as np


class Raster():
    def __init__(self, filename=None, channel=[
                 0, 1, 2], display=False, debug=False):
        """
        filename: could be filedir or path of single file
        """
        print("# ---------------------------------------------------------------------------- #")
        print("#                            TIFF process Toolkit                              #")
        print("# ---------------------------------------------------------------------------- #")
        self.display = display
        self.debug = debug
        self.channel = channel
        if not filename is None:
            if os.path.isfile(filename):
                print("# -----TIFF Class Init with :", filename)
                self.filename = filename
                self.readtif(self.filename)
        else:
            print("# -----Class TIF init without filename")

    # ---------------------------------------------------------------------------- #
    #                                     Init                                     #
    # ---------------------------------------------------------------------------- #

    def readtif(self, filename):
        self.dataset = gdal.Open(filename)  # 打开文件
        assert self.dataset is not None, "Can't Read Dataset ,Invalid tif file : " + filename
        self.width = self.dataset.RasterXSize  # 栅格矩阵的列数
        self.height = self.dataset.RasterYSize  # 栅格矩阵的行数
        self.geotransform = self.dataset.GetGeoTransform()  # 仿射矩阵
        self.projection = self.dataset.GetProjection()  # 地图投影信息
        self.image = self.dataset.ReadAsArray(0, 0, self.width, self.height)
        # print('-----Original Data Shape : ',self.image.shape)
        if 'uint8' in self.image.dtype.name:
            self.datatype = gdal.GDT_Byte
            # print('image type : uint8')
        elif 'int8' in self.image.dtype.name:
            # print('image type : int8')
            self.datatype = gdal.GDT_Byte
        elif 'int16' in self.image.dtype.name:
            # print('image type : int16')
            self.datatype = gdal.GDT_UInt16
        else:
            # print('image type : float32')
            self.datatype = gdal.GDT_Float32
        if len(self.image.shape) == 2:
            self.channel_count = 1
        if len(self.image.shape) == 3:
            self.channel_count, _, _ = self.image.shape
            if self.channel_count > 20:
                _, _, self.channel_count = self.image.shape
            else:
                self.image = self.image.transpose(1, 2, 0)
                self.image = self.image[:, :, self.channel[:]]
        if self.display:
            self.displayimagery()

    def displayimagery(self):
        self.percentage = self.fast_percentager_strentching(self.image)
        plt.imshow(self.percentage), plt.show()

    # ---------------------------------------------------------------------------- #
    #                                     Read                                     #
    # ---------------------------------------------------------------------------- #

    # ---------------------------------------------------------------------------- #
    #                         fast_percentager_strentching                         #
    # ---------------------------------------------------------------------------- #

    def fast_percentager_strentching(
            self, image=None, percentage=2, sample=10000):
        """
        Image ndarray:(W,H,C)
        Percentage N(0-100)%
        """

        assert not percentage > 100 or percentage < 0, "Invalde Percentage Value"
        print(
            "# -------------------------- percentager_strentching -------------------------")
        print(
            "# ------------------- process with percentage : ",
            percentage,
            "% ------------------")
        percentage = percentage / 100
        if isinstance(image, None):
            image = self.image
        W, H = image.shape[0], image.shape[1]
        w = np.random.randint(0, W, sample)
        h = np.random.randint(0, H, sample)
        if len(image.shape) == 3:
            points = image[w, h, :]
            point = [np.mean(channels) for channels in points]
        else:
            points = image[w, h]
            point = points
        pointset = sorted(point)
        min = int(sample * percentage)
        max = int(sample * (1 - percentage))
        min = pointset[min]
        max = pointset[max]
        image[image > max] = max
        image[image < min] = min
        image = (image - min) / (max - min)
        print("# ----- Max : ", max, " Min :    ", min, "-----")
        self.image = image
        return image

    def set(self, Data):
        """
        write a new raster file with bool type data
        """
        if len(Data.shape) == 2:
            assert not isinstance(
                Data[0, 0], bool), 'Polygonize Data ( SetRasterData(Data) ) Must be bool Ndarray,But now in ' + str(type(Data[0, 0]))
        self.RasterSet = True
        self.imageoutput = Data

    def writeimagery(self, name=None, format=["png"]):
        if name is None:
            name = self.filename + "_imagery.png"
        cv2.imwrite(name, self.image)

    def writetif(self, outputname,):
        """
        write file in tiff format
        """
        pass

    def resize_raster(self, resize_ratio=0.5):
        """
        cv2 resize image data
        6 parameter 1,5 is resolution ratio  its need /resize_ratio
        """
        size = (int(self.width * resize_ratio),
                int(self.height * resize_ratio))
        self.resizedimage = cv2.resize(
            self.image_nparray, size, interpolation=cv2.INTER_AREA)

        self.ResizeGeo = list(self.geotrans)
        print('input Geo parameter, :', self.ResizeGeo)
        self.ResizeGeo[1] = float(self.ResizeGeo[1] / resize_ratio)
        self.ResizeGeo[5] = float(self.ResizeGeo[5] / resize_ratio)
        print('resized Geo parameter ，：', self.ResizeGeo)
        self.geotrans = tuple(self.ResizeGeo)

    def writethreshold2shp(self):
        """
        Set the Boolmap(ndarray) & do polygonize in boolmap to save
        :return:
        """
        assert self.dataset is not None, 'Null dataset'
        assert self.RasterSet, 'Please Set Bool map in ndarray with SetRasterData() \n, Current output polygon src band is ' + \
            str(self.imageoutput)
        shp_name = self.out_middle_tif_name + '_polygonized.shp'
        srcband = self.dataset.GetRasterBand(1)
        maskband = None
        format = 'ESRI Shapefile'
        drv = ogr.GetDriverByName(format)
        dst_ds = drv.CreateDataSource(shp_name)
        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.outdataset.GetProjectionRef())

        dst_layer = dst_ds.CreateLayer(
            shp_name, geom_type=ogr.wkbPolygon, srs=srs)
        if (dst_layer is None):
            return 0, 0
        dst_field = dst_layer.GetLayerDefn().GetFieldIndex(shp_name)
        prog_func = gdal.TermProgress
        options = []
        result = gdal.Polygonize(srcband, maskband, dst_layer, dst_field, options,
                                 callback=prog_func)
        dst_ds = None
        print('Shapefile has write in ', shp_name)
        return shp_name

    def clear(self):
        print('-----TIF Object has been init with null')
        self.geotransform = None
        self.image = None
        self.dataset = None
        self.projection = None

    def createdataset(self, out_put_tif_name):
        driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
        self.outputdataset = driver.Create(
            out_put_tif_name,
            self.width,
            self.height,
            self.channel_count,
            self.datatype)
        self.outputdataset.SetGeoTransform(self.geotransform)  # 写入仿射变换参数
        self.outputdataset.SetProjection(self.projection)  # 写入投影
        print('Create Dataset With ', self.dataset)
        print('Create Shape is ', (self.height, self.width))

    # --------------------------------- Transform -------------------------------- #
    # ---------------------------------------------------------------------------- #
    #                                Cord Transform                                #
    # ---------------------------------------------------------------------------- #

    def getSRSPair(self):
        '''
        获得给定数据的投影参考系和地理参考系
        :param dataset: GDAL地理数据
        :return: 投影参考系和地理参考系
        '''
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(self.dataset.GetProjection())
        geosrs = prosrs.CloneGeogCS()
        return prosrs, geosrs

    def geo2lonlat(self, x, y):
        '''
        将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
        :param dataset: GDAL地理数据
        :param x: 投影坐标x
        :param y: 投影坐标y
        :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
        '''
        prosrs, geosrs = self.getSRSPair()
        ct = osr.CoordinateTransformation(prosrs, geosrs)
        coords = ct.TransformPoint(x, y)
        return coords[:2]

    def lonlat2geo(self, lon, lat):
        '''
        将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
        :param dataset: GDAL地理数据
        :param lon: 地理坐标lon经度
        :param lat: 地理坐标lat纬度
        :return: 经纬度坐标(lon, lat)对应的投影坐标
        '''
        prosrs, geosrs = self.getSRSPair()
        ct = osr.CoordinateTransformation(geosrs, prosrs)
        coords = ct.TransformPoint(lon, lat)
        
        return coords[:2]

    def imagexy2geo(self, row, col):
        '''
        根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
        :param dataset: GDAL地理数据
        :param row: 像素的行号
        :param col: 像素的列号
        :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
        '''
        trans = self.dataset.GetGeoTransform()
        px = trans[0] + col * trans[1] + row * trans[2]
        py = trans[3] + col * trans[4] + row * trans[5]
        return px, py

    def geo2imagexy(self, x, y):
        '''
        根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
        :param dataset: GDAL地理数据
        :param x: 投影或地理坐标x
        :param y: 投影或地理坐标y
        :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
        '''
        trans = self.dataset.GetGeoTransform()
        a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
        b = np.array([x - trans[0], y - trans[3]])
        return np.linalg.solve(a, b)  # numpy linalg.solve equation

    def lonlat2imagexy(self, x, y):
        x1, y1 = self.lonlat2geo(x, y)
        x2, y2 = self.geo2imagexy(x1, y1)
        return x2, y2

    def imagexy2lonlat(self, x, y):
        x1, y1 = self.imagexy2geo(x, y)
        x2, y2 = self.geo2lonlat(x1, y1)
        return x2, y2

    def getfiles_from_dir(self, dir):
        """
        return the filename list of tiff file from dir
        """
        assert not os.path.isdir(dir), "Invalid dir format" + str(dir)
        print("# -----Read Dir :", dir)
        self.files = glob.glob(os.path.join(dir, "./*.tif"))


def main():
    """
    This part will show the standard function guide.
    """


if __name__ == '__main__':
    main()
