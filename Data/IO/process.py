import multiprocessing

from downloader import downloader
from obsclient import bucket
from vector import Vector
from tqdm import tqdm
import os

import gdal


def process(VectorDataSource,
    WgsCord=(116.3, 39.9, 116.6, 39.7, 13),
    Class_key='building',
    DataSourcesType='Google China',
    DataSetName="Building_Beijing_16",
    Merge=False,
    Keep_local=False,
    remote_dataset_root="DataSets/",
    over_write=True,
    multiprocess_raster=True,
    thread_count=4,
    Nodata=0
    ):
    """
    Step I:
    Init Downlaoder,Bucket,Vector

    Step II:
    Init default vector layer
    Init area , imagery level of mission

    Step III:
    Download
    Merge(Optional)
    Rasterize

    Step IV:
    Upload to Bucket

    Last Step:
    If don't save temp dataset ,clean the cache

    """
    print("\n\n\n# ---------------------------------------------------------------------------- #")
    print("# ---------------------------------- Step I ---------------------------------- #")
    print("# ---------------------------------------------------------------------------- #")



    Download=downloader(DataSourcesType)
    Bucket=bucket()
    Vec=Vector(VectorDataSource)

    remote_metaname=remote_dataset_root+DataSetName+"/.meta"
    if not over_write:
        Bucket.check(remote_metaname)




    print("\n\n\n# ---------------------------------------------------------------------------- #")
    print("# ---------------------------------- Step II --------------------------------- #")
    print("# ---------------------------------------------------------------------------- #")

    Vec.getDefaultLayerbyName(Class_key)
    Download.add_cord(*WgsCord)
    Vec.crop_default_layer_by_rect(Download.mercator_cord)

    print("\n\n\n# ---------------------------------------------------------------------------- #")
    print("# --------------------------------- Step III --------------------------------- #")
    print("# ---------------------------------------------------------------------------- #")

    image_dir=os.path.join(DataSetName,'images/')
    targets_dir=os.path.join(DataSetName,'targets/')
    print("# ===== imagery dir :",image_dir)
    print("# ===== targets dir :",targets_dir)
    if not os.path.exists("./"+DataSetName):
        os.makedirs(image_dir)
        os.makedirs(targets_dir)
    

    local_metaname=DataSetName+"/.meta"
    with open(local_metaname,"w") as meta:
        meta.write(
            "Bucket Meta:\n"+str(Bucket.getBucketMetadata())
        )
        meta.write(
            "Vector object Meta:\n"+str(Vec.meta)
        )
    meta.close()

        
        
    bucket_imagery_root=os.path.join(remote_dataset_root,image_dir)
    bucket_targets_root=os.path.join(remote_dataset_root,targets_dir)
    bucket_description_root=os.path.join(remote_dataset_root,DataSetName+"/")
        
    print("# ===== Bucket imagery root  :",bucket_imagery_root)
    print("# ===== Bucket Targets root  :",bucket_targets_root)
    print("# ===== Bucket Description root :",bucket_description_root)
    
    Bucket.cd("DataSets")
    Bucket.ls()
    
    Download.download(output_path=image_dir)
    tiles=[i["path"] for i in Download.result]

    # Vec.generate(tiles,output_path=targets_dir)
    if multiprocess_raster:
        print("# =====Multiprocessing Rasterizeing....")
        with multiprocessing.Pool(thread_count) as pool:
            if not os.path.exists(targets_dir):
                os.makedirs(targets_dir)
            label_list=[]
            for tile in tqdm(tiles):
                try:
                    Vec.readtif(tile)
                    filename=tile.split('/')[-1]
                    path=os.path.join(targets_dir,filename)
                    label_list.append(path)
                    targetDataSet = gdal.GetDriverByName('GTiff').Create(
                    path, Vec.width, Vec.height, 1, gdal.GDT_Byte)
                    targetDataSet.SetGeoTransform(Vec.geotransform)
                    targetDataSet.SetProjection(Vec.projection)
                    band = targetDataSet.GetRasterBand(1)
                    band.SetNoDataValue(Nodata)
                    band.FlushCache()
                    # print(targetDataSet)
                    # print(Vec.defaultlayer)
                    print(os.getpid())
                    pool.apply_async(gdal.RasterizeLayer,(
                        targetDataSet,
                        [1],
                        Vec.defaultlayer,
                        )
                    )
                    targetDataSet=None

                except Exception as e:
                    print(e)


    else:
        Vec.generate(tiles,output_path=targets_dir)





    print("\n\n\n# ---------------------------------------------------------------------------- #")
    print("# ---------------------------------- Step IV --------------------------------- #")
    print("# ---------------------------------------------------------------------------- #")
    print("# ===== upload dataset meta",remote_metaname)

    Bucket.upload(
        remote_path=remote_metaname,
        local_path=local_metaname
    )
        
    ## Saveing index json file
    remote_json_path=os.path.join(bucket_description_root,Download.json_path.split('/')[-1])
    print("# ===== upload dataset description",remote_json_path)
    if not over_write:
        Bucket.check(remote_json_path)
    
    Bucket.upload(
            remote_path=remote_json_path,
            local_path=Download.json_path
        )


    print("# ===== upload imagry to bucket.....")
    
    for tile in tqdm(tiles):
        file_name=tile.split('/')[-1]
        remote_tiles=os.path.join(bucket_imagery_root,file_name)
        if not over_write:
            Bucket.check(remote_tiles)
        
        Bucket.upload(
            remote_path=remote_tiles,
            local_path=tile
        )

    print("# ===== upload target to bucket.....")

    for target in tqdm(Vec.labellist):
        file_name=target.split('/')[-1]
        remote_target=os.path.join(bucket_targets_root,file_name)
        if not over_write:
            Bucket.check(remote_target)
    
        Bucket.upload(
            remote_path=remote_target,
            local_path=target
        )
    print("# ===== uploaded bucket:")
    Bucket.ls()

    if not Keep_local:

        print("# ------------------------------- Clear-cache ------------------------------- #")
        #         cmd_rm_image="rm -rf "+image_dir
        #         cmd_rm_target="rm -rf "+targets_dir
        #         os.system(cmd_rm_image)
        #         os.system(cmd_rm_target)
        cmd="rm -rf "+DataSetName
        os.system(cmd)
        print("# -------------------------------- Clear-Done ------------------------------- #")



def main():
    vecfile="/workspace/data/osm-2017-07-03-v3.6.1-china_beijing.mbtiles"
    # macfile='/Users/tanwenxuan/workspace/Data/osm-2017-07-03-v3.6.1-china_beijing.mbtiles'
    process(vecfile,Keep_local=False,over_write=True,multiprocess_raster=False)





if __name__ == '__main__':
    main()
    