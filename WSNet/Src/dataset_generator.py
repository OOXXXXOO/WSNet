import torchvision.datasets as dataset
import torch

class DatasetGenerator():
    def __init__(self):
        print('\n\n-----Dataset Generator init-----\n\n')
    def DatasetInfo(self):
        print('dataset class on pytorch version :',torch.__version__)