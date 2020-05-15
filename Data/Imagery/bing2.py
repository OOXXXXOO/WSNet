'''
Assignment 3 : An aerial Imagery with bounding box (Bing map)
================
by Sanchuang Jiang, Yifan Xu, Yizhi Hong

Usage:
------

'''

import numpy as np
import sys
import cv2
import requests
import math
import shutil
import os

EarthRadius = 6378137;
MinLatitude = -85.05112878;
MaxLatitude = 85.05112878;
MinLongitude = -180;
MaxLongitude = 180;
Key ='AvUHS417O1zRtUa8VYSYLt9IFzubV5SZqg9VtZOWV2Spw0CNA6SAgeoelGPys8w0'
# eg. https://dev.virtualearth.net/REST/V1/Imagery/Metadata/Aerial/40.714550167322159,-74.007124900817871?zl=15&key=BingMapsKey
APIurl = 'https://dev.virtualearth.net/REST/V1/Imagery/Metadata/Aerial/'
ErrorImage = cv2.imread('error.jpeg')
ErrorSIze = os.path.getsize('error.jpeg')

class Point():
	def __init__(self,lat,lon):
		self.lat = float(lat)
		self.lon = float(lon)

	def getLat(self):
		return self.lat

	def getLon(self):
		return self.lon

	def __str__(self):
		return str((self.getLat(),self.getLon()))

	def str(self):
		return str(self.getLat()) + ',' + str(self.getLon())

# ********************************************************************
# below ref:https://msdn.microsoft.com/en-us/library/bb259689.aspx
# ********************************************************************

def clip(n,minValue,maxValue):
	return min(max(n, minValue), maxValue) 

def mapSize(level):
	return 256 << level

def GroundResolution(lat,level):
	lat = clip(lat, MinLatitude, MaxLatitude);
	return math.cos(lat * math.pi / 180) * 2 * math.pi * EarthRadius / mapSize(level)

def MapScale(lat, level, screenDpi):
	return GroundResolution(lat, level) * screenDpi / 0.0254

def LatLonToPixelXY(latitude,longitude,level):
	latitude = clip(latitude, MinLatitude, MaxLatitude)
	longitude = clip(longitude, MinLongitude, MaxLongitude)
	x = (longitude + 180) / 360
	sinLatitude = math.sin(latitude * math.pi / 180)
	y = 0.5 - math.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * math.pi)

	size = mapSize(level)
	pixelX = int(clip(x * size + 0.5, 0, size - 1))
	pixelY = int(clip(y * size + 0.5, 0, size - 1))
	return pixelX,pixelY

def PixelXYToLatLon(pixelX, pixelY, level):
	Size = mapSize(level)
	x = (clip(pixelX, 0, Size - 1) / Size) - 0.5
	y = 0.5 - (clip(pixelY, 0, Size - 1) / Size)

	latitude = 90 - 360 * math.atan(math.exp(-y * 2 * math.pi)) / math.pi
	longitude = 360 * x
	return latitude,longitude

def PixelXYToTileXY(pixelX, pixelY):
	tileX = pixelX / 256
	tileY = pixelY / 256
	return int(tileX), int(tileY)

def TileXYToPixelXY(tileX, tileY):
	pixelX = tileX * 256
	pixelY = tileY * 256
	return pixelX, pixelY

def TileXYToQuadKey(tileX, tileY, level):
	quadKey = ""
	for i in range(-level,0):
		digit = 0
		mask = 1 << (-i - 1)
		if (tileX & mask) != 0:
			digit += 1
		if (tileY & mask) != 0:
			digit += 1
			digit += 1
		quadKey += str(digit)
	return quadKey

def QuadKeyToTileXY(quadKey):

    tileX = 0
    tileY = 0
    quadKey = quadKey.split()
    level = len(quadKey)

    i = level
    try:
        while i > 0:
            mask = 1 << (i - 1)
            if quadKey[level - i] == '1':
                tileX = tileX | mask
            elif quadKey[level - i] == '2':
                tileY = tileY | mask
            elif quadKey[level - i] == '3':
                tileX = tileX | mask
                tileY = tileY | mask
            i -= 1
    except:
        print("Invalid quad key")

    return tileX, tileY

# ********************************************************************
# Above ref:https://msdn.microsoft.com/en-us/library/bb259689.aspx
# ********************************************************************

# Parsing the lat and lon to 4 points
# p1:top-left p2:top-right p3:bottom-left p4:bottom-right
def parseToPoints(ip):
	x1,y1,x2,y2 = map(float,ip.split(','))
	X1 = max(x1,x2)
	X2 = min(x1,x2)
	Y1 = min(y1,y2) 
	Y2 = max(y1,y2)
	return Point(X1,Y1),Point(X1,Y2),Point(X2,Y1),Point(X2,Y2)

# Finding the Max level of the point recursively
def getAerialMax(Point,level):
	res = requests.get(APIurl + Point.str() + "?zl=" + str(level) + "&key=" + Key).json()
	status = res['statusCode']	
	if status == 200:

		resource = res['resourceSets'][0]['resources'][0]
		exist = resource['vintageEnd'] and resource['vintageStart'] is not None

		if exist:
			imgUrl = resource['imageUrl']
			return imgUrl,level
		else:
			return getAerialMax(Point,level-1)
	else:
		return getAerialMax(Point,level-1)

def getImageUrl(quadKey):
	url = "http://ecn.t3.tiles.virtualearth.net/tiles/a" + quadKey + ".jpeg?g=6358"
	return url

def getImage(quadKey):
	url = "http://ecn.t3.tiles.virtualearth.net/tiles/a" + quadKey + ".jpeg?g=6358"
	response = requests.get(url, stream=True)
	with open('temp.jpeg', 'wb') as out_file:
	    shutil.copyfileobj(response.raw, out_file)
	del response

	img = cv2.imread('temp.jpeg')
	return img

def concatenateImg(img1Url,img2Url,ax = 1):

	response = requests.get(img1Url, stream=True)
	with open('temp1.jpeg', 'wb') as out_file:
	    shutil.copyfileobj(response.raw, out_file)
	del response

	response = requests.get(img2Url, stream=True)
	with open('temp2.jpeg', 'wb') as out_file:
	    shutil.copyfileobj(response.raw, out_file)
	del response

	img1 = cv2.imread('temp1.jpeg')
	img2 = cv2.imread('temp2.jpeg')
	vis = np.concatenate((img1, img2), axis = ax)
	return vis

### recursively create quadTree

# def addTree(T,level):
# 	if level == 0:
# 		nl = ['0','1','2','3']
# 		for n in nl:
# 			T.add_node(n)
# 		return nl
# 	else:
# 		nl = []
# 		last = addTree(T,level - 1)
# 		for i in range(4):
# 			for j in range(len(last)):
# 				T.add_edge(last[j] + str(i),last[j])
# 				nl.append(last[j] + str(i))
# 		return nl

# def createQuadtree(startKey, level):
# 	T = nx.Graph()
# 	addTree(T,level)
# 	return T

def checkError(img):
	sampleSize =  os.path.getsize('temp.jpeg')
	return True if sampleSize == ErrorSIze else False

def RefitArea(level, ps, pe):
	area = []

	print('Checking Images in level', level)

	while True:
		area = []
		startPixelX, startPixelY = LatLonToPixelXY(ps.getLat(), ps.getLon(), level)
		startTileX, startTileY = PixelXYToTileXY(startPixelX, startPixelY)
		# startQuadKey = str(TileXYToQuadKey(startTileX, startTileY, level))
		print('Top left Tile:',startTileX,startTileY)

		endPixelX, endPixelY = LatLonToPixelXY(pe.getLat(), pe.getLon(), level)
		endTileX, endTileY = PixelXYToTileXY(endPixelX, endPixelY)
		# endQuadKey = str(TileXYToQuadKey(endTileX, endTileY, level))
		print('Bottom right Tile:',endTileX,endTileY)

		
		areaList = []
		for x in range(startTileX, endTileX + 1):
			ylist = []
			for y in range(startTileY, endTileY  + 1):
				ylist.append((x,y))
				areaList.append((x,y))
			area.append(ylist)

		level_ = level
		for tile in areaList:
			qk = str(TileXYToQuadKey(tile[0], tile[1], level))
			img = getImage(qk)
			if checkError(img):
				print('level', str(level), 'Missing some of images')
				level_ = level - 1
				break

		if level_ != level:
			level = level_
			continue
		else:
			break 


	return area,level

def imageProcessing(ps, pe, psLevel, peLevel):

	level = min(psLevel,peLevel)

	print('Start at level', level)

	area,level = RefitArea(level,ps,pe)

	print('Succefully find all images in level', level)
	print('#########################################\n')

	print('Combining images...')

	imgY = []
	for j,rows in enumerate(area):
		imgX = []
		qk = str(TileXYToQuadKey(rows[0][0], rows[0][1], level))
		imgUrl = getImageUrl(qk)
		for index in range(len(rows) - 1):
			qk = str(TileXYToQuadKey(rows[index + 1][0], rows[index + 1][1], level))
			imgUrl_ = getImageUrl(qk)
			if len(imgX) != 0: 
				img = getImage(qk)
				imgX = np.concatenate((imgX,img), axis = 0)
			else:
				imgX = concatenateImg(imgUrl,imgUrl_, ax=0)

		imgY.append(imgX)
		
	imgTotal = imgY[0]
	for index in range(len(imgY) - 1):
		imgTotal =  np.concatenate((imgTotal,imgY[index + 1]), axis = 1)
		cv2.imwrite('out'+ str(index) + '.jpeg', imgY[index + 1])

	cv2.imwrite('og_result.jpeg', imgTotal)

	os.remove('temp.jpeg')
	os.remove('temp1.jpeg')
	os.remove('temp2.jpeg')

	return imgTotal,level

def cropImage(og,ps,pe,level):
	startPixelX, startPixelY = LatLonToPixelXY(ps.getLat(), ps.getLon(), level)
	endPixelX, endPixelY = LatLonToPixelXY(pe.getLat(), pe.getLon(), level)

	startTileX, startTileY = PixelXYToTileXY(startPixelX, startPixelY)
	# endTileX, endTileY = PixelXYToTileXY(endPixelX, endPixelY)

	left = startPixelX - startTileX*256
	right = endPixelX - startTileX*256

	top = startPixelY - startTileY*256
	bottom = endPixelY - startTileY*256

	img = og[top:bottom, left:right]

	return img


def main():
	if len(sys.argv)==2:
		print('Coverting the point...')
		p1,p2,p3,p4 = parseToPoints(sys.argv[1])
		print('\n#########################################\n')
		# print(p1,p2,p3,p4)
		# find the max aerial image

		print('Finding the Max level of the point recursively...')
		imgUrl1,level1 = getAerialMax(p1,18)
		imgUrl4,level4 = getAerialMax(p4,18)
		print('Higest detail level For top left point:', level1)
		print('Higest detail level For bottom right point:', level4)
		print(imgUrl1)

		print('\n#########################################\n')

		# findSubKey(p1, p4,level1, level4)
		og,level = imageProcessing(p1, p4, level1, level4)
		print('Original images Done in og_result.jpeg');


		print('\n#########################################\n')
		print('Cropping the image')
		img = cropImage(og,p1,p4,level)

		cv2.imwrite('result.jpeg', img)

		print('Croped images Done in result.jpeg');


		pass
	elif len(sys.argv) < 2:
		print('please enter coordinate. e.g: IIT campus: 41.838194,-87.629760,41.830807,-87.623366')
	else:
		print('please make sure the format is correct and no space between values eg. lat1,lon1,lat2,lon2')

if __name__ == "__main__":
	main()