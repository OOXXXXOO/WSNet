import sys
from .TIF import TIF
import ogr
import os
__version__='3.5.0'
PY2 = sys.version_info[0] == 2
class SHP(TIF):
    def __init__(self,input_shp_path=''):

        print("# ---------------------------------------------------------------------------- #")
        print("#                               Shapefile Toolkit                              #")
        print("# ---------------------------------------------------------------------------- #")


        self.Driver = ogr.GetDriverByName('ESRI Shapefile')
        TIF.__init__(self)
        if os.path.isfile(input_shp_path):
            self.Input_path=input_shp_path
            self.DataSource=self.Driver.Open(self.Input_path)
            assert self.DataSource!=None,'\n\n\nERROR-----'+str(input_shp_path)+'  --- Invalid Input Shepefile\n\n\n'
            self.Layer=self.DataSource.GetLayer()
            self.Srs=self.Layer.GetSpatialRef()
            print('-----Shapefile : ',input_shp_path,'\n-Define Done~~')
        else:
            print("-----Class SHP Init without shapefile")
    def readshp(self,shp_path):
        assert os.path.isfile(shp_path),'\n\n NULL Input Shapefile Name'
        self.Input_path = input_shp_path
        self.DataSource = self.Driver.Open(self.Input_path)
        assert self.DataSource != None, '\n\n\n~' + str(input_shp_path) + '  --- Invalid Input Shepefile\n\n\n'
        self.Layer = self.DataSource.GetLayer()
        self.Srs = self.Layer.GetSpatialRef()
        print('Shapefile : \n', input_shp_path, '\n-Read Done~~')
    def Info(self):
        print(self.DataSource)
        print(self.Layer)
        print(self.Srs)


    def statsic(self, input_tif):
        """
        The Tif file Must Mapping with Geojson
        return Stats like [(tif.pixel count ),(tif. min),(tif.max),(tif.median)]
        :param input_tif:
        :return:
        """
        self.tif=input_tif
        self.stats_ = zonal_stats(self.Input_path, input_tif, stats="count min mean max median")
        return self.stats_

    def feature_field_Defn(self,output_shp_path,featurelist=(['ID','Area','Type'])):
        if os.path.exists(output_shp_path):
            print('**********************Delete Shp Source**********************')
            self.DataSource.DeleteDataSource(output_shp_path)
        self.featureseted_shp=output_shp_path
        self.OutDataSource=self.Driver.CreateDataSource(output_shp_path)
        self.out_layer = self.OutDataSource.CreateLayer('Marked', self.Srs, geom_type=ogr.wkbPolygon)
        for feature in featurelist:
            new_field = ogr.FieldDefn(feature, ogr.OFTInteger)
            new_field.SetWidth(20)
            new_field.SetPrecision(5)
            self.out_layer.CreateField(new_field)
        



    def feature_set_from_stats(self,output_shp_path):
        if os.path.exists(output_shp_path):
            print('**********************Delete Shp Source**********************')
            self.DataSource.DeleteDataSource(output_shp_path)
        self.featureseted_shp=output_shp_path
        self.OutDataSource=self.Driver.CreateDataSource(output_shp_path)
   
   
    def getdataset(self,tif_path):
        dataset = gdal.Open(tif_path)
        return dataset

    def img_arr_to_b64(self,img_arr):
        img_pil = PIL.Image.fromarray(img_arr)
        f = BytesIO()
        img_pil.save(f, format='PNG')
        img_bin = f.getvalue()
        if hasattr(base64, 'encodebytes'):
            img_b64 = base64.encodebytes(img_bin)
        else:
            img_b64 = base64.encodestring(img_bin)
        return img_b64


    def save(self,filename, shapes, imagePath, imageData=None,
            lineColor=None, fillColor=None, otherData=None,
            flags=None):
        if imageData is not None:
            imageData = base64.b64encode(imageData).decode('utf-8')
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = []
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            lineColor=lineColor,
            fillColor=fillColor,
            imagePath=imagePath,
            imageData=imageData,
        )
        for key, value in otherData.items():
            data[key] = value
        try:
            with open(filename, 'wb' if PY2 else 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise LabelFileError(e)

    def initcordarrary(self,Type,Cordlist,line_color=None,fill_color=None):
        """
        Init shapes contents
        Use like : initlabeljson()['shapes'].append(initcordarrary(Type,Cordlist))
        :param Type: Type in Shapefile with (['Storge yard', 'Container', 'Oil Tank', 'Berth'])
        :param line_color: init by null
        :param fill_color: init by null
        :param Cordlist: like [ [ [ 114.062252545881549, 33.884823424589996 ], [ 114.063734485974663, 33.885029393359083 ],..........
        :return:
        """
        shape={"label":Type,"line_color":line_color,"fill_color":fill_color,"points":Cordlist,"shape_type":"polygon"}
        return shape

    def SetTypelist(self,TYPE):
        """
        TYPE like (['Storge yard', 'Container', 'Oil Tank', 'Berth'])
        """
        self.TYPE=TYPE
























    def Shp2LabelmeJson(self,output_GeoJson=True,
              Labelme_Json_path='./Segmentation/',
              invert2Voc=True,FieldClassName='TYPE'):
        """
        invert the Shapefile to labelme_Json
        Flow I  : Read Tif & Shapefile -> Dataset & DataSource
        Flow II : Init Null labelme_json dict
        Flow III: Get Cords(x,y) & Feature and invert longtitude,latitude to image cols,rows
        Flow IV : Append image Cords list to dict
        Flow V  : Output Labelme Json
        Flow VI : Use System Command labelme_json_to_dataset / labelme2voc.py
                # It generates:
                #   - data_dataset_voc/JPEGImages
                #   - data_dataset_voc/SegmentationClass
                #   - data_dataset_voc/SegmentationClassVisualization
        :param shp_path:
        :param tif_path:
        :param Labelme_Json_path: Output Json&jpg path
        :param invert2Dataset: invert labelme json to Dataset
        :param  TYPE like (['Storge yard', 'Container', 'Oil Tank', 'Berth'])
        :return:
        """
        print("********Start Invert :",self.Input_path,'**********')
        ##########################################################
        assert self.Input_path!='','Invaild Shapefile Name'
        in_driver = ogr.GetDriverByName('ESRI Shapefile')
        in_dataSource = in_driver.Open(self.Input_path,0)


        ##########################################################
        #Read Shp&Tif Part


        assert in_dataSource!=None,'Invaild Shapefile'
        
        in_layer = in_dataSource.GetLayer()
        assert in_layer!=None,'Can t get layer ,please check shapefile layerset'
        ##########################################################
        #GetLayer
        #Save Tif Image Data &

        shapes=[]
        print('image shape is ',self.tif.image_nparray.shape)
        filename = self.tifpath.split('/')[-1]
        for i in range(in_layer.GetFeatureCount()):
            in_Feature = in_layer.GetFeature(i)
            geom = in_Feature.GetGeometryRef()

            FieldValue=in_Feature.GetField(FieldClassName)

            Json_=geom.ExportToJson()
            print(Json_)
            # return 0

            Data=json.loads(Json_)
            print('shape type is :',Data['type'])
            assert Data['type']=='Polygon','Shape Type Must be Polygon'
            Cords=Data['coordinates'][0]
            imagecord=[]
            for point in Cords:
                if(type(point[0])==np.float):
                    x,y=self.tif.geo2imagexy(np.float(point[0]),np.float(point[1]))
                    imagecord.append([int(x),int(y)])
                # print('\n',point[0],'||',point[1],' \ntransfer to ：',x,' || ',y)
                # return 0


            feature=self.initcordarrary(self.TYPE[FieldValue],imagecord)

            shapes.append(feature)
        # print('\n\nshape is :',shapes,' process done \n\n')
        # return 0
        ####################################################################
        # shapes process done     
        if(not os.path.exists(Labelme_Json_path)):
            os.system('mkdir '+Labelme_Json_path)
        filename=filename.split('.')[-2]
        self.LabelmeJsonPath=Labelme_Json_path
        outputjsonname=Labelme_Json_path+filename+'.json'
        self.save(outputjsonname,shapes,filename+'.jpg',imageData=None,
            lineColor=(255,0,0,128),fillColor=(0,255,0,128))

        Newim = Image.fromarray(self.tif.image_nparray)
        Newim.save(Labelme_Json_path + filename + '.jpg')
        print('\n\n****json   &   image save done *****\n\n')

        if output_GeoJson:
            geojsonfilename = self.Input_path + '.geojson'
            formattransfrom_command = 'ogr2ogr -f GeoJSON ' + geojsonfilename + ' ' + self.Input_path
            print(formattransfrom_command)
            os.system(formattransfrom_command)
            print(' output geojson transfrom done ')

        if invert2Voc:
            self.Json2Voclike()

            # command='labelme_json_to_dataset '+outputjsonname
            # os.system(command)
        #
        # if invertPascalVoc:
        #     imageDataPIL.save(Labelme_Json_path+filename+'.jpg')






 def Json2Voclike(self, out_dir='./VocDataset'):
        """

        :param label_file:
        label.txt & the content like:
        __ignore__
        _background_
        Iron
        Mine
        None
        land

        The Content '__ignore__' maybe should add in file by manual
        :param in_dir:
        Dir input Data like :
        in_dir-
              |
              1.jpg
              1.json
              2.jpg
              2.json
              ....
        :param out_dir:
        output Voclike Dataset Path
        :return:
        """
        print('**********************strart voc-like  process******************')
        if os.path.exists(out_dir):
            os.removedirs(out_dir)
        os.makedirs(out_dir)
        os.makedirs(osp.join(out_dir, 'JPEGImages'))
        os.makedirs(osp.join(out_dir, 'SegmentationClass'))
        os.makedirs(osp.join(out_dir, 'SegmentationClassPNG'))
        os.makedirs(osp.join(out_dir, 'SegmentationClassVisualization'))

        class_names = []
        class_name_to_id = {}
        # for i, line in enumerate(open(labels_file).readlines()):
        #     class_id = i - 1  # starts with -1
        #     class_name = line.strip()
        #     print('current class name :', class_name)
        #     class_name_to_id[class_name] = class_id
        #     if class_id == -1:
        #         assert class_name == '__ignore__'
        #         continue
        #     elif class_id == 0:
        #         assert class_name == '_background_'
        #     class_names.append(class_name)
        assert self.TYPE!=None,'None type list Please Set First'
        class_names = tuple(self.TYPE)
        self.classlist = class_names
        self.outputdir = out_dir
        print('class_names:', class_names)

        out_class_names_file = osp.join(out_dir, 'class_names.txt')
        with open(out_class_names_file, 'w') as f:
            f.writelines('\n'.join(class_names))
        print('Saved class_names:', out_class_names_file)

        colormap = labelme.utils.label_colormap(255)
        in_dir=self.LabelmeJsonPath
        for label_file in glob.glob(osp.join(in_dir, '*.json')):
            print('Generating dataset from:', label_file)
            with open(label_file) as f:
                base = osp.splitext(osp.basename(label_file))[0]
                out_img_file = osp.join(
                    out_dir, 'JPEGImages', base + '.jpg')
                out_lbl_file = osp.join(
                    out_dir, 'SegmentationClass', base + '.npy')
                out_png_file = osp.join(
                    out_dir, 'SegmentationClassPNG', base + '.png')
                out_viz_file = osp.join(
                    out_dir, 'SegmentationClassVisualization', base + '.jpg')

                data = json.load(f)

                img_file = osp.join(osp.dirname(label_file), data['imagePath'])
                img = np.asarray(PIL.Image.open(img_file))
                PIL.Image.fromarray(img).save(out_img_file)

                lbl = labelme.utils.shapes_to_label(
                    img_shape=img.shape,
                    shapes=data['shapes'],
                    label_name_to_value=class_name_to_id,
                )
                labelme.utils.lblsave(out_png_file, lbl)

                np.save(out_lbl_file, lbl)

                viz = labelme.utils.draw_label(
                    lbl, img, class_names, colormap=colormap)
                PIL.Image.fromarray(viz).save(out_viz_file)


















def main():
    shapefile=SHP()
    shapefile.readtif("/workspace/data/clip/clip1.tif")
    shapefile.displayimagery()




if __name__ == '__main__':
    main()
    