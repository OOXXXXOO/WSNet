import torch
import torchvision
import torchvision.transforms as transforms
import math

import matplotlib.pyplot as plt
import numpy as np


from torch.autograd import  Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def load_data(DATA_PATH,BATCH_SIZE):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
            transforms.Resize(512),
            transforms.RandomSizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,])

    trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False,
                                        download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=2)

    return trainloader,testloader


## Dataloader

# functions to show an image
def imgshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()





class VGG(nn.Module):
    def __init__(self,feature,num_classes=1000,init_weights=True):
        super(VGG,self).__init__()

        self.feature=feature
        self.classfier=nn.Sequential(
            #FC-4096
            nn.Linear(512*7*7,4096),
            # nn.ReLU(inplace=True)
            nn.LeakyReLU(inplace=True),
            nn.Dropout(),

            #FC-4096
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(),

            #FC-1000
            nn.Linear(4096,num_classes),
        )
        if init_weights:
            self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            print(m,'\n')
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                print('conv2d:\n',m.weight.data)
                m.weight.data.normal_(0, math.sqrt(2. / n))
                print('Modified conv2d:\n',m.weight.data)

                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                print('BN:\n',m.weight.data)
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                print('Modified BN:\n',m.weight.data)

            elif isinstance(m, nn.Linear):
                print('Linear:\n',m.weight.data)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                print('Modified Linear:\n',m.weight.data)


    def forward(self,x):
        x=self.feature(x)
        x=x.view(x.size(0),-1)
        x=self.classfier(x)
        return x


def layerbuild(cfg,batch_normal=False,in_channels=3,layer=[]):
    """
    cfg: config dictionary 
    """

    assert cfg!=None,'Invalid CFG Info'
    for layer_config in cfg:
        if layer_config=='M':
            layer.append(nn.MaxPool2d(kernel_size=3,stride=2))

        else:
            conv2d=nn.Conv2d(in_channels,layer_config,kernel_size=3,padding=1)
            
            if batch_normal:
                layer.append(conv2d)
                layer.append(nn.BatchNorm2d(layer_config))
                layer.append(nn.ReLU(inplace=True))
            
            else:
                layer.append(conv2d)
                layer.append(nn.ReLU(inplace=True))
            in_channels=layer_config

    return nn.Sequential(*layer)







def vgg16_bn(**kwargs):
    print(cfg['D']) 
    model = VGG(layerbuild(cfg['D'], batch_normal=True), **kwargs)
    return model

def train(traindata,testdata,LearningRate=0.1):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_cuda = torch.cuda.is_available()
    global best
    criterion=nn.CrossEntropyLoss()
    model=vgg16_bn()
    model.train()
    optimizer=optim.SGD(model.parameters(),lr=LearningRate)
    
    model.cuda(device=0)

    for batch_id ,(inputs,labels) in enumerate(traindata):
        print('Batch : No .',batch_id,'\n')
        inputs, labels = inputs.cuda(), labels.cuda(async=True)
        inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
        print(inputs.shape)
        #Model INPUT Size ：
        outputs=model(inputs)

        loss=criterion(outputs,labels)
        losscpu=loss.cpu()
        print('loss',losscpu)

        # prec1,prec5=accura
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()







def main():
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    DataPath='/home/winshare/Dataset'
    Batch_Size=4
    trainloader,testloader=load_data(DataPath,Batch_Size)

    # # get some random training images
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # # show images
    # # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)),'\n')
    # imgshow(torchvision.utils.make_grid(images))
    
    # net16=vgg16_bn()
    # # print(net16)
    train(trainloader,testloader)



           

if __name__ == '__main__':
    main()
    
