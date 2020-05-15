import glob
import io
import sys
import gdal
import ogr
import gdal2xyz



"""

"""
def splitlayerbyname():
    print("# ---------------------------------------------------------------------------- #")
    print("#                                  SPLITLAYER                                  #")
    print("# ---------------------------------------------------------------------------- #")


"""
gdal_rasterize -l Beijing -a mvt_id -ts 512.0 512.0 -a_nodata 0.0 -te 12959866.260358777 4848201.814434304 12960378.261331195 4848713.815406721 -ot Float32 -of GTiff /workspace/data/Water/Beijing.geojson /tmp/processing_b877e84ad6584ef5926bcf70782a9e62/804172c8e1f749b49910d6606d960a35/OUTPUT.tif

"""

def CMDrasterizebyimagery(input_shp,tif,output,para,nodata=0,data_type="Float32",outputformat="GTiff"):
    pass
    
    # command="gdal_rasterize -l "



def RasterizeByImagery(input_shp,tif,output,para,nodata=0,data_type="Float32",outputformat="GTiff"):
    dataset=gdal.Open(input_shp)