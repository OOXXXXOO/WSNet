import sys
# from Raster import Raster
from raster import Raster
import ogr
import os
import gdal
import numpy as np
from tqdm import tqdm
import json
import multiprocessing
__version__ = '3.5.0'
PY2 = sys.version_info[0] == 2


class Vector(Raster):
    def __init__(self, input_shp_path=''):

        print("# ---------------------------------------------------------------------------- #")
        print("#                                Vector Toolkit                                #")
        print("# ---------------------------------------------------------------------------- #")
        Raster.__init__(self)
        self.readshp(input_shp_path)
        self.ExportLayer = None

    def readshp(self, input_shp_path):
        if os.path.isfile(input_shp_path):
            filetype = input_shp_path.split(".")[-1]
            gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
            gdal.SetConfigOption("SHAPE_ENCODING", "")
            ogr.RegisterAll()
            self.DriveDict = {
                'shp': ogr.GetDriverByName('ESRI Shapefile'),
                'SHP': ogr.GetDriverByName('ESRI Shapefile'),
                'Geojson': ogr.GetDriverByName('GeoJSON'),
                'geojson': ogr.GetDriverByName('GeoJSON'),
                'pbf': ogr.GetDriverByName('MVT'),
                'mbtiles': ogr.GetDriverByName('MBTiles')
            }
            if filetype in self.DriveDict.keys():
                print("# -----Valid vector format :", filetype)
                self.Driver = self.DriveDict[filetype]

            self.Input_path = input_shp_path
            self.DataSource = self.Driver.Open(self.Input_path)
            self.meta = self.DataSource.GetMetadata()
            
            print("\n# ----------------------------- Meta Information ----------------------------- #")
            self.print_dict(self.meta)

            print("# ----------------------------- Meta Information ----------------------------- #\n")
            self.Description = self.DataSource.GetDescription()
            print('# -----Description : ', self.Description)
            assert self.DataSource is not None, '\n\n\nERROR-----' + \
                str(input_shp_path) + '  --- Invalid Input Shepefile\n\n\n'
            self.LayerCount = self.DataSource.GetLayerCount()
            print("# -----LayerCount:", self.LayerCount)
            self.LayerDict = {}
            for i in range(self.LayerCount):
                self.Layer = self.DataSource.GetLayer(i)
                print(
                    "# -----Layer :",
                    i,
                    " LayerName : ",
                    self.Layer.GetName(),
                    self.Layer.GetGeometryColumn())
                self.LayerDict[self.Layer.GetName()] = self.Layer
            
            
            # print("# ---------------------------- Layer dict by name ---------------------------- #")

            # self.print_dict(self.LayerDict)

            # print("# ---------------------------- Layer dict by name ---------------------------- #")
            
            self.Srs = self.Layer.GetSpatialRef()
            self.Extent=self.Layer.GetExtent()
            print("# -----Extent:",self.Extent)
            print("# -----Alread Load:", input_shp_path)
            print(
                "# -------------------------------- DEFINE DONE ------------------------------- #")

        else:
            print("# -----Class SHP Init without shapefile")

    def getDefaultLayerbyName(self, name):
        """
        para:name string of layer name
        """
        self.defaultlayer = self.LayerDict[name]
        print("# -----Set Default Layer |",name,"| : ",self.defaultlayer)
        return self.LayerDict[name]

    def Info(self):
        print(self.DataSource)
        print(self.Layer)
        print(self.Srs)

    def getbounding(self):
        adfGeoTransform = self.geotransform
        x1 = adfGeoTransform[0] + 0 * \
            adfGeoTransform[1] + 0 * adfGeoTransform[2]
        y1 = adfGeoTransform[3] + 0 * \
            adfGeoTransform[4] + 0 * adfGeoTransform[5]
        x2 = adfGeoTransform[0] + self.width * \
            adfGeoTransform[1] + self.height * adfGeoTransform[2]
        y2 = adfGeoTransform[3] + self.width * \
            adfGeoTransform[4] + self.height * adfGeoTransform[5]
        self.bounding = (x1, y1, x2, y2)
        return self.bounding

    def getdataset(self, tif_path):
        self.tif_path = tif_path
        self.dataset = gdal.Open(tif_path)
        print('\n# -----Dataset:', self.dataset)
        return self.dataset

    def SaveTo(self, name):
        filetype = name.split(".")[-1]
        if filetype in self.DriveDict.keys():
            out = self.DriveDict[filetype].CopyDataSource(
                self.DataSource, name)
            out = None

    def SaveVectorByLayerName(self, LayerName, outputname, format="GeoJSON"):
        """
        format:
            * ESRI Shapefile
            * GeoJSON
        """
        self.ExportLayer = self.LayerDict[LayerName]
        print("# ------------------Start Create Layer",
              LayerName, ' ---------------------- #')
        self.ExportDriver = ogr.GetDriverByName(format)

        if os.path.isfile(outputname):
            self.ExportDriver.DeleteDataSource(outputname)

        ExportResources = self.ExportDriver.CreateDataSource(outputname)
        print("-----ExportResources:", ExportResources)
        assert not ExportResources is None, "Invalid Export Resources ,Please check the path of output"
        ExportLayerTemp = ExportResources.CreateLayer(
            LayerName, geom_type=ogr.wkbPolygon)
        print("-----Data Resources Create Done")

        # Add input Layer Fields to the output Layer
        inLayerDefn = self.ExportLayer.GetLayerDefn()
        for i in range(0, inLayerDefn.GetFieldCount()):
            fieldDefn = inLayerDefn.GetFieldDefn(i)
            ExportLayerTemp.CreateField(fieldDefn)

        # Get the output Layer's Feature Definition
        outLayerDefn = ExportLayerTemp.GetLayerDefn()

        # Add features to the ouput Layer
        print("-----Start Loop Process")
        for i in tqdm(range(0, self.ExportLayer.GetFeatureCount())):
            # Get the input Feature
            inFeature = self.ExportLayer.GetFeature(i)
            # Create output Feature
            outFeature = ogr.Feature(outLayerDefn)
            # Add field values from input Layer
            for i in range(0, outLayerDefn.GetFieldCount()):
                outFeature.SetField(
                    outLayerDefn.GetFieldDefn(i).GetNameRef(),
                    inFeature.GetField(i))
            # Set geometry as centroid
            geom = inFeature.GetGeometryRef()
            inFeature = None
            centroid = geom.Centroid()
            outFeature.SetGeometry(centroid)
            # Add new feature to output Layer
            ExportLayerTemp.CreateFeature(outFeature)
            outFeature = None

        print("# -----LayerCopyDone,", LayerName)
        ExportResources.SetMetadata(self.meta)
        print("# -----Set Meta Done")
        ExportResources.SetDescription(self.Description)
        print("# -----Set Description Done")
        ExportResources.Destroy()
        print("# --------------------------------- Save Done -------------------------------- #")

        # Save and close DataSources
        inDataSource = None
        outDataSource = None

    def crop_layer_by_polygon(self, vector):
        self.ExportLayer.SetSpatialFilter()

    def crop_layer_by_rect(self, rect):
        assert self.ExportLayer is None, 'Invalid Export Layer'
        self.ExportLayer.SetSpatialFilterRect()

    
    def crop_default_layer_by_rect(self, rect):
        print("# -----Set filter Rect:",rect)
        self.defaultlayer.SetSpatialFilterRect(*rect)
    
    def print_dict(self,d,n=0):
        length=69
        for k,v in d.items():
            # print ('\t'*n)
            if type(v)==type({}):
                print("%s : {" % k)
                self.print_dict(v,n+1)
            else:
                
                strl=len(str(k))+len(str(v))
                space=length-strl
                if strl>length:                    
                    v=str(v)[:space]
                print("# -----%s : %s" % (k,v)+" "*space+"#")
        if n!=0:
            print('\t'*(n-1)+ '}')
    
    # def Shp2LabelmeJson(self, output_GeoJson=True,
    #                     Labelme_Json_path='./Segmentation/',
    #                     invert2Voc=True, FieldClassName='TYPE'):
    #     """
    #     invert the Shapefile to labelme_Json
    #     Flow I  : Read Tif & Shapefile -> Dataset & DataSource
    #     Flow II : Init Null labelme_json dict
    #     Flow III: Get Cords(x,y) & Feature and invert longtitude,latitude to image cols,rows
    #     Flow IV : Append image Cords list to dict
    #     Flow V  : Output Labelme Json
    #     Flow VI : Use System Command labelme_json_to_dataset / labelme2voc.py
    #             # It generates:
    #             #   - data_dataset_voc/JPEGImages
    #             #   - data_dataset_voc/SegmentationClass
    #             #   - data_dataset_voc/SegmentationClassVisualization
    #     :param shp_path:
    #     :param tif_path:
    #     :param Labelme_Json_path: Output Json&jpg path
    #     :param invert2Dataset: invert labelme json to Dataset
    #     :param  TYPE like (['Storge yard', 'Container', 'Oil Tank', 'Berth'])
    #     :return:
    #     """

    #     print('-----feature count :', self.Layer.GetFeatureCount())
    #     print("-----SpatialFilter :", self.Layer.GetSpatialFilter())

    #     print("-----", self.Layer.GetLayerDefn())

    #     print("-----", self.Layer.GetFeatureCount())

    #     print("-----", self.Layer.GetExtent())

    #     x1, y1, x2, y2 = self.Layer.GetExtent()
    #     x1, y1 = self.geo2imagexy(x1, y1)
    #     x2, y2 = self.geo2imagexy(x2, y2)
    #     print("-----Image:", x1, y1, x2, y2)
    #     width = abs(x2 - x1)
    #     height = abs(y2 - y1)
    #     print("-----Whole Image shape:", height, width)

    #     print('-----', self.Layer[0])

    #     print("-----", self.Layer.GetSpatialRef())
    #     exit(0)

    #     # for i in tqdm(range(self.Layer.GetFeatureCount())):
    #     #     in_Feature = self.Layer.GetFeature(i)
    #     #     geom = in_Feature.GetGeometryRef()
    #     #     print(type(geom))
    #     #     print(geom)
    #     # FieldValue=in_Feature.GetField(FieldClassName)
    #     # geodict=geom.ExportToJson()
    #     # geodict=

    #     # geodict=json.loads(geodict)

    #     # feature=self.initcordarrary(self.TYPE[FieldValue],imagecord)

    #     # shapes.append(feature)
    #     # print('\n\nshape is :',shapes,' process done \n\n')
    #     # return 0
    #     ####################################################################
    #     # shapes process done
    #     if(not os.path.exists(Labelme_Json_path)):
    #         os.system('mkdir ' + Labelme_Json_path)
    #     filename = filename.split('.')[-2]
    #     self.LabelmeJsonPath = Labelme_Json_path
    #     outputjsonname = Labelme_Json_path + filename + '.json'
    #     self.save(outputjsonname, shapes, filename + '.jpg', imageData=None,
    #               lineColor=(255, 0, 0, 128), fillColor=(0, 255, 0, 128))

    #     Newim = Image.fromarray(self.tif.image_nparray)
    #     Newim.save(Labelme_Json_path + filename + '.jpg')
    #     print('\n\n****json   &   image save done *****\n\n')

    #     if output_GeoJson:
    #         geojsonfilename = self.Input_path + '.geojson'
    #         formattransfrom_command = 'ogr2ogr -f GeoJSON ' + \
    #             geojsonfilename + ' ' + self.Input_path
    #         print(formattransfrom_command)
    #         os.system(formattransfrom_command)
    #         print(' output geojson transfrom done ')

    #     if invert2Voc:
    #         self.Json2Voclike()

    #         # command='labelme_json_to_dataset '+outputjsonname
    #         # os.system(command)
    #     #
    #     # if invertPascalVoc:
    #     #     imageDataPIL.save(Labelme_Json_path+filename+'.jpg')

    def Rasterize(self, outputname, Nodata=0):

        targetDataSet = gdal.GetDriverByName('GTiff').Create(
            outputname, self.width, self.height, 1, gdal.GDT_Byte)
        targetDataSet.SetGeoTransform(self.geotransform)
        targetDataSet.SetProjection(self.projection)
        band = targetDataSet.GetRasterBand(1)
        band.SetNoDataValue(Nodata)
        band.FlushCache()
        gdal.RasterizeLayer(
            targetDataSet,
            [1],
            self.defaultlayer,
            )
        targetDataSet = None
        
    def generate(self,tiles,output_path="./label"):
        print('# ===== Start Generate.....')
        """
        修订  把Rasterize同样使用线程分解的可能性
        """

        self.labellist=[]
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for tile in tqdm(tiles):
            self.readtif(tile)
            filename=tile.split('/')[-1]
            path=os.path.join(output_path,filename)
            self.labellist.append(path)
            self.Rasterize(path)
        return self.labellist










    
    def reset_layer(self):
        self.Layer.ResetReading()

    # def Json2Voclike(self, out_dir='./VocDataset'):
    #     """

    #     :param label_file:
    #     label.txt & the content like:
    #     __ignore__
    #     _background_
    #     Iron
    #     Mine
    #     None
    #     land

    #     The Content '__ignore__' maybe should add in file by manual
    #     :param in_dir:
    #     Dir input Data like :
    #     in_dir-
    #         |
    #         1.jpg
    #         1.json
    #         2.jpg
    #         2.json
    #         ....
    #     :param out_dir:
    #     output Voclike Dataset Path
    #     :return:
    #     """
    #     print('**********************strart voc-like  process******************')
    #     if os.path.exists(out_dir):
    #         os.removedirs(out_dir)
    #     os.makedirs(out_dir)
    #     os.makedirs(os.path.join(out_dir, 'JPEGImages'))
    #     os.makedirs(os.path.join(out_dir, 'SegmentationClass'))
    #     os.makedirs(os.path.join(out_dir, 'SegmentationClassPNG'))
    #     os.makedirs(os.path.join(out_dir, 'SegmentationClassVisualization'))

    #     class_names = []
    #     class_name_to_id = {}
    #     # for i, line in enumerate(open(labels_file).readlines()):
    #     #     class_id = i - 1  # starts with -1
    #     #     class_name = line.strip()
    #     #     print('current class name :', class_name)
    #     #     class_name_to_id[class_name] = class_id
    #     #     if class_id == -1:
    #     #         assert class_name == '__ignore__'
    #     #         continue
    #     #     elif class_id == 0:
    #     #         assert class_name == '_background_'
    #     #     class_names.append(class_name)
    #     assert self.TYPE is not None, 'None type list Please Set First'
    #     class_names = tuple(self.TYPE)
    #     self.classlist = class_names
    #     self.outputdir = out_dir
    #     print('class_names:', class_names)

    #     out_class_names_file = os.path.join(out_dir, 'class_names.txt')
    #     with open(out_class_names_file, 'w') as f:
    #         f.writelines('\n'.join(class_names))
    #     print('Saved class_names:', out_class_names_file)

    #     colormap = labelme.utils.label_colormap(255)
    #     in_dir = self.LabelmeJsonPath
    #     for label_file in glob.glob(os.path.join(in_dir, '*.json')):
    #         print('Generating dataset from:', label_file)
    #         with open(label_file) as f:
    #             base = os.path.splitext(os.path.basename(label_file))[0]
    #             out_img_file = os.path.join(
    #                 out_dir, 'JPEGImages', base + '.jpg')
    #             out_lbl_file = os.path.join(
    #                 out_dir, 'SegmentationClass', base + '.npy')
    #             out_png_file = os.path.join(
    #                 out_dir, 'SegmentationClassPNG', base + '.png')
    #             out_viz_file = os.path.join(
    #                 out_dir, 'SegmentationClassVisualization', base + '.jpg')

    #             data = json.load(f)

    #             img_file = os.path.join(os.path.dirname(label_file), data['imagePath'])
    #             img = np.asarray(PIL.Image.open(img_file))
    #             PIL.Image.fromarray(img).save(out_img_file)

    #             lbl = labelme.utils.shapes_to_label(
    #                 img_shape=img.shape,
    #                 shapes=data['shapes'],
    #                 label_name_to_value=class_name_to_id,
    #             )
    #             labelme.utils.lblsave(out_png_file, lbl)

    #             np.save(out_lbl_file, lbl)

    #             viz = labelme.utils.draw_label(
    #                 lbl, img, class_names, colormap=colormap)
    #             PIL.Image.fromarray(viz).save(out_viz_file)


def main():
    Vector = SHP("/workspace/data/Water/Beijing.geojson")
    # Vector=SHP("/workspace/SQCV/Data/IO/waterchina.geojson")

    # ------------------------------ Rasterize Demo ------------------------------ #

    # Vector=SHP("/home/winshare/Downloads/2017-07-03_asia_china.mbtiles").SaveTo("CHINA.shp")
    # Vector.SaveVectorByLayerName('water',"waterchina.shp",format="ESRI Shapefile")

    import glob

    tiflist = glob.glob("/workspace/data/Water/GoogleBeijing/*.tif")
    # tiflist.extend(glob.glob("/workspace/data/Water/BingBeijing/*.tif"))

    # Experiment--Geojson
    # ----------------------------------- Speed ---------------------------------- #
    #                               十万样本时间
    # CHINA             34          94  小时
    # Beijing           1/13        120 分钟
    # MBtilesFilter     3           8.3 小时

    # Experiment--Shapefile
    # ----------------------------------- Speed ---------------------------------- #

    print(len(tiflist))
    for file in tqdm(tiflist):
        Vector.readtif(file)
        Vector.getDefaultLayerbyName("bui")

        filename = file.split('/')[-1]
        path = '/workspace/data/Water/label/' + filename
        Vector.Rasterize(path)


if __name__ == '__main__':
    main()
