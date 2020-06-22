from obs import ObsClient



class bucket():
    def __init__(self,
    access_key_id=None,
    secret_access_key=None,
    server=None,
    bucketName=None
    ):
    # 创建ObsClient实例
        self.base_folder="/"
        if access_key_id==None or secret_access_key==None or server== None:
            
            self.access_key_id='ISSSVUVTXQWXSCPKR23N'
            self.secret_access_key='IYZbHMxJss3vXsoi9pqArIySf205lPcoISmm6ReJ'
            self.server='http://obs.cn-north-4.myhuaweicloud.com'

            self.obsClient = ObsClient(
            access_key_id=self.access_key_id,    
            secret_access_key=self.secret_access_key,    
            server=self.server
            )

        else:
            self.obsClient = ObsClient(
            access_key_id=access_key_id,    
            secret_access_key=secret_access_key,    
            server=server
            )

        if bucketName==None:
            self.bucketName='obs-tq-dataset'
            self.bucketClient=self.obsClient.bucketClient(self.bucketName)
        else:
            self.bucketName=bucketName
            self.bucketClient=self.obsClient.bucketClient(bucketName)

        print("# ---------------------------------------------------------------------------- #")
        print("#                                Bucket ToolKit                                #")
        print("# ---------------------------------------------------------------------------- #")       
        print("# ----access key (AK) : ",self.access_key_id)
        print("# ----secret key (SK): ",self.secret_access_key)
        print("# ----server : ",self.server)
        print("# ----bucket name : ",self.bucketName)
        print("# ----root : ",self.base_folder)
        print("# ---------------------------------------------------------------------------- #")       
        


    def getBucketMetadata(self):
        print('Getting bucket metadata\n')
        #resp = obsClient.getBucketMetadata(bucketName, origin='http://www.b.com', requestHeaders='Authorization1')
        resp = self.bucketClient.getBucketMetadata(origin='http://www.b.com', requestHeaders='Authorization1')
        print('storageClass:', resp.body.storageClass)
        print('accessContorlAllowOrigin:', resp.body.accessContorlAllowOrigin)
        print('accessContorlMaxAge:', resp.body.accessContorlMaxAge)
        print('accessContorlExposeHeaders:', resp.body.accessContorlExposeHeaders)
        print('accessContorlAllowMethods:', resp.body.accessContorlAllowMethods)
        print('accessContorlAllowHeaders:', resp.body.accessContorlAllowHeaders)
        print('Deleting bucket CORS\n')
        resp = self.bucketClient.deleteBucketCors()
        print('status'  + str(resp.status))
        return resp

    def upload(self,remote_path,local_path):
        self.obsClient.putFile(self.bucketName,remote_path, local_path)
        # print("# ===== Uploading ",local_path," ===to : ",remote_path)

    
    def download(self,key,download):
        # print("# ===== Downloading ",key," === to :",download)
        self.obsClient.getObject(self.bucketName, key, downloadPath=download)


    def cd(self,folder_key):
        self.base_folder=folder_key
        print("# ===== Base Folder",self.base_folder)


    def delete(self,key):
        print('# ===== Deleting object ' +key + '\n')
        self.obsClient.deleteObject(self.bucketName, key)

    def check(self,key):
        """
        The Sync will overwrite by default. We need check
        """
        assert not self.obsClient.getObject(self.bucketName,key)["status"]<300,"\n# ===== ERROR : \n# ===== bucket : ({bucketname})\n# ===== key : ({key}) & local upload flow try to overwrite same key".format(bucketname=self.bucketName,key=key)


    def mkdir(self,dir):
        pass

    def display(self,dir):
        splited=dir.split("/")
        sp="      "*len(splited)+"└-----|"
        return sp+dir





    def ls(self,show_item_count=10):
        print("# ===== list ({path}): ".format(path=self.base_folder))
        # resp = self.obsClient.listObjects(self.bucketName)

        if self.base_folder=="/":
            resp = self.obsClient.listObjects(self.bucketName)
            
        else:
            resp = self.obsClient.listObjects(self.bucketName,self.base_folder)
        keylist=[]
        print("# ===== object count : ",len(resp.body.contents))
        for content in resp.body.contents[:show_item_count]:
            keylist.append(content.key)
            # print('   └-----|' + content.key )
            print(self.display(content.key))
        return keylist

def main():
    Bucket=bucket()
    # Bucket.upload("/port.vin","./Data/IO/port.vin")
    # Bucket.download("/port.vin","./hash.vin")
    # Bucket.getBucketMetadata()
    Bucket.cd("Description/")
    Bucket.ls()
    Bucket.delete("Description/")




if __name__ == '__main__':
    main()
    