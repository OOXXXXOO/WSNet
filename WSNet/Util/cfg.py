import os
import sys
import numpy as np
import json
from Instace import *
class cfg(Instance):
    def __init__(self,missionType,root):
        print('process type with :',missionType)

        if missionType=='Segmentation':
            print('root is ',root)
            assert os.path.exists(root),"Invaild root path for Segmentation"
            imagepath=os.path.join(root,'image')
            labelpath = os.path.join(root, 'label')




        elif missionType=='Detection':
            print('root is ', root)
            assert os.path.exists(root), "Invaild root path for Detection"
            imagepath=os.path.join(root,'image')
            labelpath = os.path.join(root, 'label')


        else:
            print('Invaild Mission Type,Please Check Missiontype')
            sys.exit()
            
        # return super().__init__(*args, **kwargs)
    def read(self,config_json):

CFG=cfg(missionType='Segmentation',root='./')
CFG2=cfg('Detection','./')