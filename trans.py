import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import torchvision

# img='DocResources/quick.jpg'
# imgg=Image.open(img)
# print(imgg.size)
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
# # model.eval()

# # predictions = model(x)
# # print(predictions)
# model.to('cuda:0')
# model.train()
# params = [p for p in model.parameters() if p.requires_grad]
# print(params)
# optim=torch.optim.SGD(params,lr=0.001,momentum=0.9,weight_decay=1e-4)
# lossfunction=nn.CrossEntropyLoss()


# print(optim)

# print(lossfunction)


# ########One batch
# x = torch.rand(2,3, 300, 400)
# x=x.to('cuda:0')
# output=model(x)
# print(output)

# a=[[1,2,3,4,5]]
# a=torch.tensor(a)
# print(a)
class A():
    def __init__(self):
        print('A')


    def hhh(self):
        D=A()
        D.hhh()


class B():
    def __init__(self,A):
        print('B')

E=A()
E.hhh()

