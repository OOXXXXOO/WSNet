import os
import sys
import numpy as np
import json



class cfg():
    """
    Mission index :
    0 -> Detection Mission
    1 -> Segmentation Mission
    2 -> Mask Mission
    others -> Invalid Mission


    """
    def __init__(self,missionType,root='./'):
        """

        :param missionType:
        Mission index :
        0 -> Detection Mission
        1 -> Segmentation Mission
        2 -> Mask Mission
        others -> Invalid Mission
        :param root:
        root path of config file
        """
        super(cfg,self).__init__()
        assert missionType.type()!=int or missionType>=3,'mode index must be interger(0-3) & '
        print('process type with :',self.MissionType[missionType])

        print('root is ',root)
        imagepath = os.path.join(root, 'image')
        labelpath = os.path.join(root, 'label')
        if missionType=='1':

            assert os.path.exists(root),"Invaild root path for Segmentation"


        elif missionType=='0':

            assert os.path.exists(root), "Invaild root path for Detection"



        elif missionType=='2':
            assert os.path.exists(root), "Invaild root path for Instance Segmentation Mission"


        else:
            print('Invaild Mission Type,Please Check Missiontype')
            sys.exit()
            
        # return super().__init__(*args, **kwargs)
    def read(self,config_json):
        pass



    def generatetemplate(self,jsonfile='./config.json'):
        pass
        # instance={}
        # instance['instance id']=self.instance_id
        # net={'backbone':self.backbone,''}
        # net['backbone']=
        # instance['Net']=self.
        # instance['mode']=self.
#
# CFG=cfg(missionType='Segmentation',root='./')
# CFG2=cfg('Detection','./')