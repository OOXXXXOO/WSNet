from downloader import downloader
from obsclient import bucket
from vector import Vector
from tqdm import tqdm
import os


def process(VectorDataSource,
    WgsCord=(116.3, 39.9, 116.6, 39.7, 13),
    Class_key='building',
    DataSourcesType='Google China',
    DataSetName="Building_Beijing",
    Merge=False,
    Keep_local=False
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

    print("\n# ---------------------------------- Step I ---------------------------------- #")
    Download=downloader(DataSourcesType)
    Bucket=bucket()
    Vec=Vector(VectorDataSource)

    print("\n\n# ---------------------------------- Step II --------------------------------- #")
    Vec.getDefaultLayerbyName(Class_key)
    Download.add_cord(*WgsCord)
    Vec.crop_default_layer_by_rect(Download.mercator_cord)

    print("\n\n\n# --------------------------------- Step III --------------------------------- #")
    image_dir=os.path.join(DataSetName,'images/')
    targets_dir=os.path.join(DataSetName,'targets/')
    print("# ===== imagery dir :",image_dir)
    print("# ===== targets dir :",targets_dir)
    if not os.path.exists("./"+DataSetName):
        os.makedirs(image_dir)
        os.makedirs(targets_dir)
        
        
    bucket_imagery_root=os.path.join("DataSets/",image_dir)
    bucket_targets_root=os.path.join("DataSets/",targets_dir)
    bucket_description_root="Description/"
    
    print("# ===== Bucket imagery root  :",bucket_imagery_root)
    print("# ===== Bucket Targets root  :",bucket_targets_root)
    Bucket.cd("DataSets")
    Bucket.ls()
    
    Download.download(output_path=image_dir)
    tiles=[i["path"] for i in Download.result]

    Vec.generate(tiles,output_path=targets_dir)

    print("\n\n\n# ---------------------------------- Step IV --------------------------------- #")
        

        
    ## Saveing index json file
    Bucket.upload(
            remote_path=os.path.join(bucket_description_root,Download.json_path.split('/')[-1]),
            local_path=Download.json_path
        )
    
    for tile in tqdm(tiles):
        file_name=tile.split('/')[-1]
        Bucket.upload(
            remote_path=os.path.join(bucket_imagery_root,file_name),
            local_path=tile
        )

    for target in tqdm(Vec.labellist):
        file_name=target.split('/')[-1]
        Bucket.upload(
            remote_path=os.path.join(bucket_targets_root,file_name),
            local_path=target
        )
    if not Keep_local:

        print("\n\n\n# ------------------------------- Clearn cache ------------------------------- #")
        #         cmd_rm_image="rm -rf "+image_dir
        #         cmd_rm_target="rm -rf "+targets_dir
        #         os.system(cmd_rm_image)
        #         os.system(cmd_rm_target)
        cmd="rm -rf "+DataSetName
        print("# -------------------------------- Clearn Done ------------------------------- #")



def main():
    vecfile="/workspace/data/osm-2017-07-03-v3.6.1-china_beijing.mbtiles"
    process(vecfile)





if __name__ == '__main__':
    main()
    