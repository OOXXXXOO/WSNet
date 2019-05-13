import cv2
import torch.nn as nn
import os
import argparse








def main():
    print('\n\n------Basic DataSet Builder-----\n\n')
    
    parser = argparse.ArgumentParser(description=
    "------PyTorch Dataset Building Tool Help Notes------")
 
    parser.add_argument('--imagedir',type=str,default='./dataset',help='must input image dir (default is ./dataset )')

    parser.add_argument('--labeldir',type=str,default='./dataset/label',help='must input label dir (default is ./dataset/label)')

    parser.add_argument('--ratio',type=float,default=0.8,help='ratio of train data & val data (default=0.8)')

    # parser.add_argument('--')

    args = parser.parse_args()
    print('backbone :',args.backbone)



if __name__ == '__main__':
    main()
    