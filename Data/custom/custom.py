import time
import math
from tqdm import tqdm
import tifffile as tif
import glob
import os
class Customdataset():
    def __init__(self,TIF_Path,Label_Path,Viz_Path=None,labels=None,Fast=False):
        """
        Satellite imagery dataset generate
            The satellite imagery has huge pixel count like 10000^2 pixels
            So the single imagery can't be train & inference directly. we should random resample
            the imagery to build dataset.
            In the /Src/Utils/DataToolkit/labelme2coco.py we could transfom&visualize the labelme label dataset

        para:
            Fast : bool type of generate npy/npz fast dataset

        """
        TIFS=sorted(glob.glob(os.path.join(TIF_Path,"*.tif")))
        Labels=sorted(glob.glob(os.path.join(Label_Path,"*.png")))

        for index in tqdm(range(len(TIFS))):
            tif=TIFS[index]
            label=Labels[index]
            


    def random_generate(self,patch_size=32,sample_ratio=0.3,npy=None,csv=None):
        
        self.datapath=os.path.join(self.root,'Data')
        self.labelpath=os.path.join(self.root,'Label')
    
        if not os.path.exists(self.datapath):
            print('create dataset folder in',self.datapath)
            os.makedirs(self.datapath)
        if not os.path.exists(self.labelpath):
            print('create labelset folder in',self.labelpath)
            os.makedirs(self.labelpath)
 
    
        patch=[]
        # with open(self.csv, 'a') as csvfile:
        H=self.inputdata.shape[0]-patch_size
        W=self.inputdata.shape[1]-patch_size
        num=int((H*W)/(patch_size**2*sample_ratio))
        print('Data Length is ,',num)
        if csv!=None:
            self.csv=os.path.join(self.root,csv)
            if not os.path.exists(self.csv):
                print('create index csv in',self.csv)
                os.system('touch '+self.csv)
        if npy!=None:
            self.npy=os.path.join(self.root,npy)
            if not os.path.exists(self.npy):
                print('create index csv in',self.npy)
                os.system('touch '+self.npy)
            print(self.cordcentre)
            for i in tqdm(range(num)):
                x=np.random.randint(0, H)
                y=np.random.randint(0, W)
                image=self.inputdata[x:x+patch_size,y:y+patch_size]/self.max_height
                label=self.labeldata[x:x+patch_size,y:y+patch_size]
                if label.sum()>0:
                    label=1
                else:
                    label=0
                distence=math.sqrt((x-self.cordcentre[0])**2+(y-self.cordcentre[1])**2)/((math.sqrt(2)*(max(H,W))))
                distence=float(distence)
                # print((image,label, self.intensity[x,y]/10,x-self.cordcentre[0]/W,y-self.cordcentre[1]/H,distence))
                patch.append((image,
                             label,
                             self.intensity[x,y]/10,
                             float((x-self.cordcentre[0])/W),
                             float((y-self.cordcentre[1])/H),
                             distence))