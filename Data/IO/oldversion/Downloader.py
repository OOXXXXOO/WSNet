import numpy as np
import math
import gdal
from math import floor, pi, log, tan, atan, exp
from threading import Thread
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

Tilesize = 256

# WGS-84 to Web Mercator

def wgs_to_mercator(x, y):
    y = 85.0511287798 if y > 85.0511287798 else y
    y = -85.0511287798 if y < -85.0511287798 else y
    x2 = x * 20037508.34 / 180
    y2 = log(tan((90 + y) * pi / 360)) / (pi / 180)
    y2 = y2 * 20037508.34 / 180
    return x2, y2

# Web Mercator to WGS-84

def mercator_to_wgs(x, y):
    x2 = x / 20037508.34 * 180
    y2 = y / 20037508.34 * 180
    y2 = 180 / pi * (2 * atan(exp(y2 * pi / 180)) - pi / 2)
    return x2, y2
# --------------------------------------------------------------------------------------

# -----------------Interchange between GCJ-02 to WGS-84-------------------
# All public geographic data in mainland China need to be encrypted with GCJ-02, introducing random bias
# This part of the code is used to remove the bias


def transformLat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * \
        y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 *
            math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 *
            math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 *
            math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def transformLon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + \
        0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 *
            math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 *
            math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 *
            math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def delta(lat, lon):
    '''
    Krasovsky 1940
    //
    // a = 6378245.0, 1/f = 298.3
    // b = a * (1 - f)
    // ee = (a^2 - b^2) / a^2;
    '''
    a = 6378245.0  # a: Projection factor of satellite ellipsoidal coordinates projected onto a flat map coordinate system
    ee = 0.00669342162296594323  # ee: Eccentricity of ellipsoid
    dLat = transformLat(lon - 105.0, lat - 35.0)
    dLon = transformLon(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * math.pi
    magic = math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = math.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * math.pi)
    dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * math.pi)
    return {'lat': dLat, 'lon': dLon}


def outOfChina(lat, lon):
    if (lon < 72.004 or lon > 137.8347):
        return True
    if (lat < 0.8293 or lat > 55.8271):
        return True
    return False


def gcj_to_wgs(gcjLon, gcjLat):
    if outOfChina(gcjLat, gcjLon):
        return (gcjLon, gcjLat)
    d = delta(gcjLat, gcjLon)
    return (gcjLon - d["lon"], gcjLat - d["lat"])


def wgs_to_gcj(wgsLon, wgsLat):
    if outOfChina(wgsLat, wgsLon):
        return wgsLon, wgsLat
    d = delta(wgsLat, wgsLon)
    return wgsLon + d["lon"], wgsLat + d["lat"]
# --------------------------------------------------------------

# ---------------------------------------------------------
# Get tile coordinates in Google Maps based on latitude and longitude of WGS-84


def wgs_to_tile(j, w, z):
    '''
    Get google-style tile cooridinate from geographical coordinate
    j : Longittude
    w : Latitude
    z : zoom
    '''
    def isnum(x): return isinstance(x, int) or isinstance(x, float)
    if not(isnum(j) and isnum(w)):
        raise TypeError("j and w must be int or float!")

    if not isinstance(z, int) or z < 0 or z > 22:
        raise TypeError("z must be int and between 0 to 22.")

    if j < 0:
        j = 180 + j
    else:
        j += 180
    j /= 360  # make j to (0,1)

    w = 85.0511287798 if w > 85.0511287798 else w
    w = -85.0511287798 if w < -85.0511287798 else w
    w = log(tan((90 + w) * pi / 360)) / (pi / 180)
    w /= 180  # make w to (-1,1)
    w = 1 - (w + 1) / 2  # make w to (0,1) and left top is 0-point

    num = 2**z
    x = floor(j * num)
    y = floor(w * num)
    return x, y


def pixls_to_mercator(zb):
    # Get the web Mercator projection coordinates of the four corners of the
    # area according to the four corner coordinates of the tile
    inx, iny = zb["LT"]  # left top
    inx2, iny2 = zb["RB"]  # right bottom
    length = 20037508.3427892
    sum = 2**zb["z"]
    LTx = inx / sum * length * 2 - length
    LTy = -(iny / sum * length * 2) + length

    RBx = (inx2 + 1) / sum * length * 2 - length
    RBy = -((iny2 + 1) / sum * length * 2) + length

    # LT=left top,RB=right buttom
    # Returns the projected coordinates of the four corners
    res = {'LT': (LTx, LTy), 'RB': (RBx, RBy),
           'LB': (LTx, RBy), 'RT': (RBx, LTy)}
    return res


def tile_to_pixls(zb):
    # Tile coordinates are converted to pixel coordinates of the four corners
    out = {}
    width = (zb["RT"][0] - zb["LT"][0] + 1) * Tilesize
    height = (zb["LB"][1] - zb["LT"][1] + 1) * Tilesize
    out["LT"] = (0, 0)
    out["RT"] = (width, 0)
    out["LB"] = (0, -height)
    out["RB"] = (width, -height)
    return out


class Downloader(Thread):
    # multiple threads downloader
    def __init__(self, index, count, urls, datas):
        # index represents the number of threads
        # count represents the total number of threads
        # urls represents the list of URLs nedd to be downloaded
        # datas represents the list of data need to be returned.
        super().__init__()
        self.urls = urls
        self.datas = datas
        self.index = index
        self.count = count

    def download(self, url):
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
                return data
        raise Exception("Bad network link.")

    def run(self):
        for i, url in enumerate(self.urls):
            if i % self.count != self.index:
                continue
            self.datas[i] = self.download(url)
# ---------------------------------------------------------

# ---------------------------------------------------------


def getExtent(x1, y1, x2, y2, z, source="Google China"):
    pos1x, pos1y = wgs_to_tile(x1, y1, z)
    pos2x, pos2y = wgs_to_tile(x2, y2, z)
    Xframe = pixls_to_mercator({"LT": (pos1x, pos1y), "RT": (
        pos2x, pos1y), "LB": (pos1x, pos2y), "RB": (pos2x, pos2y), "z": z})
    for i in ["LT", "LB", "RT", "RB"]:
        Xframe[i] = mercator_to_wgs(*Xframe[i])
    if source == "Google":
        pass
    elif source == "Google China":
        for i in ["LT", "LB", "RT", "RB"]:
            Xframe[i] = gcj_to_wgs(*Xframe[i])
    else:
        pass
        # raise Exception("Invalid argument: source.")
    return Xframe


def saveTiff(r, g, b, gt, filePath):
    driver = gdal.GetDriverByName('GTiff')
    # Create a 3-band dataset
    outRaster = driver.Create(filePath,
        r.shape[1],
        r.shape[0],
        3,
        gdal.GDT_Byte)
    outRaster.SetGeoTransform(gt)
    try:
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
    except Exception as e:
        print(e)
    outRaster.GetRasterBand(1).WriteArray(r)
    outRaster.GetRasterBand(2).WriteArray(g)
    outRaster.GetRasterBand(3).WriteArray(b)
    outRaster.FlushCache()
    outRaster = None
    # print("Image Saved")
# ---------------------------------------------------------


# ---------------------------------------------------------
MAP_URLS = {
    "Google": "http://mts0.googleapis.com/vt?lyrs={style}&x={x}&y={y}&z={z}",
    "Google China": "http://mt2.google.cn/vt/lyrs={style}&hl=zh-CN&gl=CN&src=app&x={x}&y={y}&z={z}",

    "Google Maps": "https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
    "Google Satellite": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "Google Terrain": "https://mt1.google.com/vt/lyrs=t&x={x}&y={y}&z={z}",
    "Google Terrain Hybrid": "https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}",
    "Google Satellite Hybrid": "https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}",

    "Stamen Terrain": "http://tile.stamen.com/terrain/{z}/{x}/{y}.png",
    "Stamen Toner": "http://tile.stamen.com/toner/{z}/{x}/{y}.png",
    "Stamen Toner Light": "http://tile.stamen.com/toner-lite/{z}/{x}/{y}.png",
    "Stamen Watercolor": "http://tile.stamen.com/watercolor/{z}/{x}/{y}.jpg",

    "Wikimedia Map": "https://maps.wikimedia.org/osm-intl/{z}/{x}/{y}.png",
    "Wikimedia Hike Bike Map": "http://tiles.wmflabs.org/hikebike/{z}/{x}/{y}.png",

    "Esri Boundaries Places": "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    "Esri Gray (dark)": "http://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    "Esri Gray (light)": "http://services.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
    "Esri National Geographic": "http://services.arcgisonline.com/ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}",
    "Esri Ocean": "https://services.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}",
    "Esri Satellite": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "Esri Standard": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    "Esri Terrain": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Terrain_Base/MapServer/tile/{z}/{y}/{x}",
    "Esri Transportation": "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
    "Esri Topo World": "http://services.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",

    "OpenStreetMap Standard": "http://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "OpenStreetMap H.O.T.": "http://tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
    "OpenStreetMap Monochrome": "http://tiles.wmflabs.org/bw-mapnik/{z}/{x}/{y}.png",
    "OpenTopoMap": "https://tile.opentopomap.org/{z}/{x}/{y}.png",

    "Strava All": "https://heatmap-external-b.strava.com/tiles/all/bluered/{z}/{x}/{y}.png",
    "Strava Run": "https://heatmap-external-b.strava.com/tiles/run/bluered/{z}/{x}/{y}.png?v=19",

    "Open Weather Map Temperature": "http://tile.openweathermap.org/map/temp_new/{z}/{x}/{y}.png?APPID=1c3e4ef8e25596946ee1f3846b53218a",
    "Open Weather Map Clouds": "http://tile.openweathermap.org/map/clouds_new/{z}/{x}/{y}.png?APPID=ef3c5137f6c31db50c4c6f1ce4e7e9dd",
    "Open Weather Map Wind Speed": "http://tile.openweathermap.org/map/wind_new/{z}/{x}/{y}.png?APPID=f9d0069aa69438d52276ae25c1ee9893",

    "CartoDb Dark Matter": "http://basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png",
    "CartoDb Positron": "http://basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png",

    "Bing VirtualEarth": "http://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1"

}


def TileXYToQuadKey(tileX, tileY, level):
    quadKey = ""
    for i in range(-level, 0):
        digit = 0
        mask = 1 << (-i - 1)
        if (tileX & mask) != 0:
            digit += 1
        if (tileY & mask) != 0:
            digit += 1
            digit += 1
        quadKey += str(digit)
    return quadKey


def get_url(source, x, y, z, style):
    if source == 'Google China':
        url = MAP_URLS["Google China"].format(x=x, y=y, z=z, style=style)
    elif source == "Bing VirtualEarth":
        quadkey = TileXYToQuadKey(x, y, z)
        url = MAP_URLS["Bing VirtualEarth"].format(q=quadkey)

    elif source == 'Google':
        url = MAP_URLS["Google"].format(x=x, y=y, z=z, style=style)
    else:
        url = MAP_URLS[source].format(x=x, y=y, z=z)
        # pass
        # raise Exception("Unknown Map Source ! ")
    return url


def get_urls(x1, y1, x2, y2, z, source, style):
    pos1x, pos1y = wgs_to_tile(x1, y1, z)
    pos2x, pos2y = wgs_to_tile(x2, y2, z)
    lenx = pos2x - pos1x + 1
    leny = pos2y - pos1y + 1
    print("-----Total tiles number：{x} X {y}".format(x=lenx, y=leny))
    urls = [
        get_url(
            source,
            i,
            j,
            z,
            style) for j in range(pos1y,pos1y +leny) for i in range(pos1x,pos1x +lenx)
            ]
    return urls
# ---------------------------------------------------------

# ---------------------------------------------------------



def merge_tiles(datas, x1, y1, x2, y2, z):
    pos1x, pos1y = wgs_to_tile(x1, y1, z)
    pos2x, pos2y = wgs_to_tile(x2, y2, z)
    lenx = pos2x - pos1x + 1
    leny = pos2y - pos1y + 1
    outpic = pil.new('RGBA', (lenx * Tilesize, leny * Tilesize))
    for i, data in enumerate(datas):
        picio = io.BytesIO(data)
        small_pic = pil.open(picio)
        y, x = i // lenx, i % lenx
        outpic.paste(small_pic, (x * Tilesize, y * Tilesize))
    print('-----Tiles merge completed')
    return outpic


def download_tiles(urls, multi=8):
    url_len = len(urls)
    datas = [None] * url_len
    if multi < 1 or multi > 20 or not isinstance(multi, int):
        raise Exception(
            "multi of Downloader shuold be int and between 1 to 20.")
    tasks = [Downloader(i, multi, urls, datas) for i in range(multi)]
    for i in tasks:
        i.start()

    for i in tasks:
        i.join()

    print(multiprocessing.current_process())
    print("-----Buffer DataSize:", sys.getsizeof(datas))
    return datas
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

    def addcord(self, left, top, right, bottom, zoom, style='s'):
        self.left, self.top, self.right, self.bottom, self.zoom = left, top, right, bottom, zoom
        self.extent = getExtent(
            self.left,
            self.top,
            self.right,
            self.bottom,
            self.zoom,
            self.server)
        # Get the urls of all tiles in the extent
        urls = get_urls(left, top, right, bottom, zoom, self.server, style)
        # Group URLs based on the number of CPU cores to achieve roughly equal
        # amounts of tasks
        self.urls_group = [urls[i:i + math.ceil(len(urls) / multiprocessing.cpu_count())]
                           for i in range(0, len(urls), math.ceil(len(urls) / multiprocessing.cpu_count()))]

        return self.urls_group

    def download(self):
        # Each set of URLs corresponds to a process for downloading tile maps
        print('Tiles downloading......')
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        results = pool.map(download_tiles, self.urls_group)
        pool.close()
        pool.join()
        self.result = [x for j in results for x in j]

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

    Google = DOWNLOADER("Bing VirtualEarth")
    Google.addcord(116.3, 39.9, 116.6, 39.7, 13)
    Google.download()
    tiles = Google.savetiles(path="./Image", format="tif")
    from Vector import Vector
    Beijing=Vector("/workspace/data/Water/Beijing.geojson")
    Beijing.getDefaultLayerbyName("Beijing")
    Beijing.generate(tiles,output_path='./Label')


if __name__ == '__main__':
    main()
