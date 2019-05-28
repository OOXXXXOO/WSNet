import torch.nn as nn
print('reference sucess')









class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,
    inplane,
    planes,
    stride=1,
    downsample=None,
    groups=1,
    base_width=64,
    dilation=1,
    norm_layer=None):
        super(BasicBlock,self).__init__()
        if norm_layer is None:
            norm_layer=nn.BatchNorm2d
        if groups!=1 or base_width!=64:
            raise ValueError('Basic Block only support groups =1 & base_width=64')
        return super().__init__()














class resnet(nn.Module):
    def __init__(self):
        return super().__init__()