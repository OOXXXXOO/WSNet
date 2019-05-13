import os
import sys
import numpy as np


class cfg:
    def __init__(self,missionType,root):
        print('process type with :',missionType)
        if missionType=='Segmentation':
            print('root is ',root)
            assert not os.path.exists(root),"Invaild root path for Segmentation"
            continue
        if missionType=='Detection':
            print('root is ', root)
            assert not os.path.exists(root), "Invaild root path for Detection"
            continue
        else:
            print('Invaild Mission Type,Please Check Missiontype')
            sys.exit()
            
        # return super().__init__(*args, **kwargs)

CFG=cfg(missionType='Segmentation',root='LL')