import torch.nn as nn
import torch
import torchvision
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import math

class Bottleneck(nn.Module):
    expansion=4
    def __init__(self,inplanes,planes,baseWidth,cardinality,stride=1,dawnsample=None):
        """
        :param inplanes: 输入通道维数
        :param planes: 输出通道纬数
        :param baseWidth: 基本宽度
        :param cardinality: 卷积组数量
        :param stride: 步副
        :param dawnsample:是否下采样
        """
        super(Bottleneck,self).__init__()
        D=int(math.floor(planes*(basewidth/64)))
        C=cardinality

