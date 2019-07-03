import torch
import torch.nn as nn

class net1(nn.Module):
    def __init__(self):
        super(net1,self).__init__()

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1000, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Linear(256, 2),
        )

    def forward(self,x):
        x=self.classifier(x)
        return x

class net2(nn.Module):
    def __init__(self):
        super(net2,self).__init__()
        self.FC1=nn.Linear(1000,2,bias=True)
        
    def forward(self,x):
        x=self.FC1(x)

        return x
a=torch.randn((1,1000))
Net1=net1()
Net2=net2()
b=Net1(a)
c=Net2(a)
print(a,'\n',b,'\n',c)
