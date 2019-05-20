import cv2
import torch.nn as nn
import os
import argparse
from Instance import *

class datasetbuilder(Instance):
    def __init__(self,root):
        print('******* dataset builder ******')
        assert not os.path.exists(root),'Invalid Root Path'
        assert not os.path.exists(os.path.join(root,'image')),'cant find image folder'
        assert not os.path.exists(os.path.join(root,'label')),'cant find label folder'
        self.root=root
        self.imagefolder=os.path.join(root,'image')
        self.labelfolder=os.path.join(root,'image')

        







def main():
    print('\n\n------Basic DataSet Builder-----\n\n')
    
    parser = argparse.ArgumentParser(description=
    "------PyTorch Dataset Building Tool Help Notes------")
 
    parser.add_argument('--root',type=str,default='./dataset',help='must input image dir (default is ./dataset )')

    parser.add_argument('--labeldir',type=str,default='./dataset/label',help='must input label dir (default is ./dataset/label)')

    parser.add_argument('--ratio',type=float,default=0.8,help='ratio of train data & val data (default=0.8)')

    # parser.add_argument('--')

    args = parser.parse_args()
    print('backbone :',args.root)
    """
    Process flow:
    1.Traversal image folder & index image file & label file
     
    """
    dataset=datasetbuilder(args.root)




if __name__ == '__main__':
    main()
    