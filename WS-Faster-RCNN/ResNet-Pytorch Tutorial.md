# Pytorch 复现笔记-ResNet（ResNext）

这个系列主旨从基础出发，顺着复现文章的角度，学习优秀的复现Trick，了解这些神经网络背后的原理。

- [Pytorch 复现笔记-ResNet（ResNext）](#pytorch-%E5%A4%8D%E7%8E%B0%E7%AC%94%E8%AE%B0-resnetresnext)
  - [所用到的高级特性：](#%E6%89%80%E7%94%A8%E5%88%B0%E7%9A%84%E9%AB%98%E7%BA%A7%E7%89%B9%E6%80%A7)
    - [1.新式类与经典类](#1%E6%96%B0%E5%BC%8F%E7%B1%BB%E4%B8%8E%E7%BB%8F%E5%85%B8%E7%B1%BB)
    - [2.Super关键字和绑定方法，非绑定方法](#2super%E5%85%B3%E9%94%AE%E5%AD%97%E5%92%8C%E7%BB%91%E5%AE%9A%E6%96%B9%E6%B3%95%E9%9D%9E%E7%BB%91%E5%AE%9A%E6%96%B9%E6%B3%95)
- [数据IO，处理，清洗](#%E6%95%B0%E6%8D%AEio%E5%A4%84%E7%90%86%E6%B8%85%E6%B4%97)
- [从VGG看基础共用网络结构的实现](#%E4%BB%8Evgg%E7%9C%8B%E5%9F%BA%E7%A1%80%E5%85%B1%E7%94%A8%E7%BD%91%E7%BB%9C%E7%BB%93%E6%9E%84%E7%9A%84%E5%AE%9E%E7%8E%B0)
      - [这个实现的实例中，有非常多的Trick 我们需要一一讲解一下](#%E8%BF%99%E4%B8%AA%E5%AE%9E%E7%8E%B0%E7%9A%84%E5%AE%9E%E4%BE%8B%E4%B8%AD%E6%9C%89%E9%9D%9E%E5%B8%B8%E5%A4%9A%E7%9A%84trick-%E6%88%91%E4%BB%AC%E9%9C%80%E8%A6%81%E4%B8%80%E4%B8%80%E8%AE%B2%E8%A7%A3%E4%B8%80%E4%B8%8B)
  - [要点解析](#%E8%A6%81%E7%82%B9%E8%A7%A3%E6%9E%90)
      - [nn.Module](#nnmodule)
      - [Batch Normalization](#batch-normalization)
      - [Dropout](#dropout)
      - [Initialize_weights](#initializeweights)
      - [isinstance](#isinstance)
      - [forward](#forward)
      - [tensor.view](#tensorview)
      - [transfrom](#transfrom)
      - [](#)
  - [卷积操作与卷基层](#%E5%8D%B7%E7%A7%AF%E6%93%8D%E4%BD%9C%E4%B8%8E%E5%8D%B7%E5%9F%BA%E5%B1%82)
  - [池化操作与池化层](#%E6%B1%A0%E5%8C%96%E6%93%8D%E4%BD%9C%E4%B8%8E%E6%B1%A0%E5%8C%96%E5%B1%82)
  - [Linear](#linear)
  - [BasicBlock & Bottleneck](#basicblock--bottleneck)
  - [Softmax](#softmax)
  - [Loss Cross Entropy](#loss-cross-entropy)
  - [Train & Test](#train--test)
  - [ResNet](#resnet)
- [WS-Faster-RCNN](#ws-faster-rcnn)



## 所用到的高级特性：
### 1.新式类与经典类
```python
class Bottleneck:
#经典类
class Bottleneck(nn.Module):
#新式类
```

区别在于是否集成object类， 修复了经典类多继承的bug
下面我们着重说一下多继承的bug 如图：

![](https://images2015.cnblogs.com/blog/1091061/201704/1091061-20170430134623303-568049990.png)

BC 为A的子类， D为BC的子类 ，A中有save方法，C对其进行了重写

在经典类中 调用D的save方法 搜索按深度优先 路径B-A-C， 执行的为A中save 显然不合理
在新式类的 调用D的save方法 搜索按广度优先 路径B-C-A， 执行的为C中save
### 2.Super关键字和绑定方法，非绑定方法
super 是用来解决多重继承问题的，直接用类名调用父类方法在使用单继承的时候没问题，但是如果使用多继承，会涉及到查找顺序（MRO）、重复调用（钻石继承）等种种问题。总之前人留下的经验就是：保持一致性。要不全部用类名调用父类，要不就全部用 super，不要一半一半。

当我们定义了一个父类A，再用B继承A，在B的构造函数中调用A的构造，则会遍历整个类定义把所有非绑定方法的类目全部替换

```python
class Base:
    def __init__(self, value):
        self.value = value
 
 
class A(Base):
    def __init__(self, value):
        Base.__init__(self, value)
        self.value *= 2
 
 
class B(Base):
    def __init__(self, value):
        Base.__init__(self, value)
        self.value += 3
 
 
class C(A, B):
    def __init__(self, value):
        A.__init__(self, value)
        B.__init__(self, value)
 
foo = C(5)
print(foo.value)

```
当我们这样定义，按照我们的思路输入5结果应该是5*2+3=13才对
但是结果却是8
当调用B的时候会再次调用Base将self.value刷新为5
所以我们尝试用super关键字对继承流程进行修饰
```python
class Base:
    def __init__(self, value):
        self.value = value
 
 
class A(Base):
    def __init__(self, value):
        super(A, self).__init__(value)
        self.value *= 2
 
 
class B(Base):
    def __init__(self, value):
        super(B, self).__init__(value)
        self.value += 3
 
 
class C(A, B):
    def __init__(self, value):
        super(C, self).__init__(value)
 
foo = C(5)
print(foo.value)

```
借助super关键字我们能按照c3算法对超类初始化流程进行规范，但是以上的代码执行的结果依然不对
执行流程变为了MRO顺序则变成了base->B->A于是变成了（5+3）*2=16
```python
class Base:
    def __init__(self, value):
        self.value = value
 
 
class A(Base):
    def __init__(self, value):
        super(A, self).__init__(value * 2)
 
 
class B(Base):
    def __init__(self, value):
        super(B, self).__init__(value + 3)
 
 
class C(A, B):
    def __init__(self, value):
        super(C, self).__init__(value)
 
foo = C(5)
print(foo.value)
 
13

```
类中的方法要么是绑定给对象使用，要么是绑定给类使用，那么有没有不绑定给两者使用的函数？

　　答案：当然有，python给我们提供了@staticmethod，可以解除绑定关系，将一个类中的方法，变为一个普通函数。

　　下面，我们来看看代码示例：
```python
import hashlib
import time
class MySQL:

    def __init__(self,host,port):
        self.id=self.create_id()
        self.host=host
        self.port=port
    
    @staticmethod
    
    def create_id(): #就是一个普通工具
        m=hashlib.md5(str(time.clock()).encode('utf-8'))
        return m.hexdigest()


print(MySQL.create_id) #<function MySQL.create_id at 0x0000000001E6B9D8> #查看结果为普通函数
conn=MySQL('127.0.0.1',3306)
print(conn.create_id) #<function MySQL.create_id at 0x00000000026FB9D8> #查看结果为普通函数
```


总体用到的高级特性大概就这几种  下面开始正式的复现



# 数据IO，处理，清洗

公共格式数据集的参考意义不大，我们需要对自己的数据及生成方式进行定义，先来看看官方的自定义数据类：

```python
class ImageFolder(data.Dataset):
    """默认图像数据目录结构
    root
    .
    ├──dog
    |   ├──001.png
    |   ├──002.png
    |   └──...
    └──cat  
    |   ├──001.png
    |   ├──002.png
    |   └──...
    └──...
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        classes, class_to_idx = find_classes(root)
        imgs = make_dataset(root, class_to_idx)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        index (int): Index
	Returns:tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imgs)
```


————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————



# 从VGG看基础共用网络结构的实现

![](https://img-blog.csdn.net/20180205192403250?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvZGNybWc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)


我们可以对照这个图来说明一个基础CNN的结构，值得注意的是VGG的复现工作大多数来源于根据字典来配置网络结构的一个版本如下
```python

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        """
        着重注意一点
        这里的features就是CNN所提取出的特征向量
        也就是make_layers最后返回的神经网络结构(*layers)被包装在nn.Sequential里面
        """
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

#网络构建实例的方法demo
def vgg16_bn(**kwargs):
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model

def vgg16(**kwargs):
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


```
上面代码中利用ABDE来区分网络结构以及利用是否进行BN进行网络构建
其对应的配置信息也就如下图：

![table](https://img-blog.csdn.net/20180820164714340?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2FkbWludGFu/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

#### 这个实现的实例中，有非常多的Trick  我们需要一一讲解一下
比如根据配置字典构造网络结构的这个方法

```python

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

```
这个部分有个很容易误导人的写法就是
```python
a=[]
b='1'
a+=[b]
```
其实如果从易读性角度考虑
```python
a.append(b)
```
这种写法可能比较合适

因此这个方法可以改写成：
```python

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

```
改写完成之后我们实例化一个VGG16BN网络，然后打印（注释添加对应原文的参数）：


```python
VGG(
  (feature): Sequential(
    #conv3-64
    (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace)
    #conv3-64
    (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace)
    #maxpool
    (6): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    #conv3-128
    (7): Conv2d(64, 128, kernel_size=(3, 3),
     stride=(1, 1), padding=(1, 1))
    (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (9): ReLU(inplace)
    #conv3-128
    (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (12): ReLU(inplace)
    #maxpool
    (13): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    #conv3-256
    (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (16): ReLU(inplace)
    #conv3-256
    (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (19): ReLU(inplace)
    #conv3-256
    (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (22): ReLU(inplace)
    #maxpool
    (23): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    #conv3-512
    (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (26): ReLU(inplace)
    #conv3-512
    (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (29): ReLU(inplace)
    #conv3-512
    (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (32): ReLU(inplace)
    #maxpool
    (33): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
    #conv3-512
    (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (36): ReLU(inplace)
    #conv3-512
    (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (39): ReLU(inplace)
    #conv3-512
    (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (42): ReLU(inplace)
    #maxpool
    (43): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  )


  (classfier): Sequential(
    
    (0): Linear(in_features=25088, out_features=4096, bias=True)
    
    (1): LeakyReLU(negative_slope=0.01, inplace)
    
    (2): Dropout(p=0.5)
    
    (3): Linear(in_features=4096, out_features=4096, bias=True)
    
    (4): ReLU(inplace)
    
    (5): Dropout(p=0.5)
    
    (6): Linear(in_features=4096, out_features=1000, bias=True)
  )
)

```
到了这里，就算基本完成了网络的定义工作  因此在正式开始讲解神经网络的基础结构之前先来看看这段代码中体现出来的需要注意的部分，我们需要对几个点进行解析
## 要点解析
#### nn.Module
```python
class VGG(nn.Module):
    def __init__(self,feature,num_classes=1000,init_weights=True):
        super(VGG,self).__init__()
```
**class torch.nn.Module**
所有网络的基类。

你的模型也应该继承这个类。并且调用Module的析构函数



#### Batch Normalization
　　BN的基本思想其实相当直观：

因为深层神经网络在做非线性变换前的激活输入值（就是那个x=WU+B，U是输入）随着网络深度加深或者在训练过程中，其分布逐渐发生偏移或者变动，之所以训练收敛慢，一般是整体分布逐渐往非线性函数的取值区间的上下限两端靠近（对于Sigmoid函数来说，意味着激活输入值WU+B是大的负值或正值），所以这导致反向传播时低层神经网络的梯度消失，这是训练深层神经网络收敛越来越慢的本质原因，

而BN就是通过一定的规范化手段，把每层神经网络任意神经元这个输入值的分布强行拉回到均值为0方差为1的标准正态分布，其实就是把越来越偏的分布强制拉回比较标准的分布，

这样使得激活输入值落在非线性函数对输入比较敏感的区域，这样输入的小变化就会导致损失函数较大的变化，意思是这样让梯度变大，避免梯度消失问题产生，而且梯度变大意味着学习收敛速度快，能大大加快训练速度。


![](https://images2017.cnblogs.com/blog/1093303/201802/1093303-20180219084749642-1647361064.png)

如图具体步骤：
* 求每一个训练批次数据的均值


* 求每一个训练批次数据的方差


* 使用求得的均值和方差对该批次的训练数据做归一化，获得0-1分布。其中ε是为了避免除数为0时所使用的微小正数。


* 尺度变换和偏移：将xi乘以γ调整数值大小，再加上β增加偏移后得到yi，这里的γ是尺度因子，β是平移因子。这一步是BN的精髓，由于归一化后的xi基本会被限制在正态分布下，使得网络的表达能力下降。为解决该问题，我们引入两个新的参数：γ,β。 γ和β是在训练时网络自己学习得到的。

得到的结果可视化就如图：

![](https://images2017.cnblogs.com/blog/1093303/201802/1093303-20180219084810095-616879424.png)


![](https://images2017.cnblogs.com/blog/1093303/201802/1093303-20180219084820533-1615172856.png)

在原理层面上我们使用
图把求解的过程展开：

![process](https://img-blog.csdn.net/20160318140751330)

前向传播和反向传播的过程如上图所示，我们尝试用numpy来模拟这个过程

```python
def batchnorm_forward(x, gamma, beta, eps):

  N, D = x.shape
  #为了后向传播求导方便，这里都是分步进行的
  #step1: 计算均值
  mu = 1./N * np.sum(x, axis = 0)

  #step2: 减均值
  xmu = x - mu

  #step3: 计算方差
  sq = xmu ** 2
  var = 1./N * np.sum(sq, axis = 0)

  #step4: 计算x^的分母项
  sqrtvar = np.sqrt(var + eps)
  ivar = 1./sqrtvar

  #step5: normalization->x^
  xhat = xmu * ivar

  #step6: scale and shift
  gammax = gamma * xhat
  out = gammax + beta

  #存储中间变量
  cache =  (xhat,gamma,xmu,ivar,sqrtvar,var,eps)

  return out, cache

```
反向传播的求导过程则要复杂很多

```python
def batchnorm_backward(dout, cache):

  #解压中间变量
  xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache

  N,D = dout.shape

  #step6
  dbeta = np.sum(dout, axis=0)
  dgammax = dout
  dgamma = np.sum(dgammax*xhat, axis=0)
  dxhat = dgammax * gamma

  #step5
  divar = np.sum(dxhat*xmu, axis=0)
  dxmu1 = dxhat * ivar #注意这是xmu的一个支路

  #step4
  dsqrtvar = -1. /(sqrtvar**2) * divar
  dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

  #step3
  dsq = 1. /N * np.ones((N,D)) * dvar
  dxmu2 = 2 * xmu * dsq #注意这是xmu的第二个支路

  #step2
  dx1 = (dxmu1 + dxmu2) 注意这是x的一个支路


  #step1
  dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
  dx2 = 1. /N * np.ones((N,D)) * dmu 注意这是x的第二个支路

  #step0 done!
  dx = dx1 + dx2

  return dx, dgamma, dbeta

```





















#### Dropout
在机器学习的模型中，如果模型的参数太多，而训练样本又太少，训练出来的模型很容易产生过拟合的现象。在训练神经网络的时候经常会遇到过拟合的问题，过拟合具体表现在：模型在训练数据上损失函数较小，预测准确率较高；但是在测试数据上损失函数比较大，预测准确率较低。

过拟合是很多机器学习的通病。如果模型过拟合，那么得到的模型几乎不能用。为了解决过拟合问题，一般会采用模型集成的方法，即训练多个模型进行组合。此时，训练模型费时就成为一个很大的问题，不仅训练多个模型费时，测试多个模型也是很费时。

综上所述，训练深度神经网络的时候，总是会遇到两大缺点：

（1）容易过拟合

（2）费时

Dropout可以比较有效的缓解过拟合的发生，在一定程度上达到正则化的效果。
其第一次出现在AlexNet上

![](https://img-blog.csdn.net/20180619185225799?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3Byb2dyYW1fZGV2ZWxvcGVy/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

其原理就是在前向传播的时候让某个神经元的激活值以一定的概率p停止工作，让一半的特征检测器（隐层节点）停止工作（值为0），这样做可以减少隐层节点之间的相互作用。
举例来说，如果大量的隐层节点强烈的依靠另一个隐层节点才能发生作用，那特征就局限在局部了，Dropout能够使模型的泛化能力增强。如同上图所示

1. 为什么说Dropout可以解决过拟合？
（1）取平均的作用： 先回到标准的模型即没有dropout，我们用相同的训练数据去训练5个不同的神经网络，一般会得到5个不同的结果，此时我们可以采用 “5个结果取均值”或者“多数取胜的投票策略”去决定最终结果。例如3个网络判断结果为数字9,那么很有可能真正的结果就是数字9，其它两个网络给出了错误结果。这种“综合起来取平均”的策略通常可以有效防止过拟合问题。因为不同的网络可能产生不同的过拟合，取平均则有可能让一些“相反的”拟合互相抵消。dropout掉不同的隐藏神经元就类似在训练不同的网络，随机删掉一半隐藏神经元导致网络结构已经不同，整个dropout过程就相当于对很多个不同的神经网络取平均。而不同的网络产生不同的过拟合，一些互为“反向”的拟合相互抵消就可以达到整体上减少过拟合。

（2）减少神经元之间复杂的共适应关系： 因为dropout程序导致两个神经元不一定每次都在一个dropout网络中出现。这样权值的更新不再依赖于有固定关系的隐含节点的共同作用，阻止了某些特征仅仅在其它特定特征下才有效果的情况 。迫使网络去学习更加鲁棒的特征 ，这些特征在其它的神经元的随机子集中也存在。换句话说假如我们的神经网络是在做出某种预测，它不应该对一些特定的线索片段太过敏感，即使丢失特定的线索，它也应该可以从众多其它线索中学习一些共同的特征。从这个角度看dropout就有点像L1，L2正则，减少权重使得网络对丢失特定神经元连接的鲁棒性提高。

我们的神经网络计算的推到通常是由

![](https://img-blog.csdn.net/20151227174051344?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

在Dropout之后就变成了

![](https://img-blog.csdn.net/20151227174102788?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

这个过程可以使用numpy来进行实现模拟

```python
import numpy as np

#dropout函数的实现
def dropout(x, level):
    if level < 0. or level >= 1:#level是概率值，必须在0~1之间
        raise Exception('Dropout level must be in interval [0, 1[.')
    retain_prob = 1. - level
    #我们通过binomial函数，生成与x一样的维数向量。binomial函数就像抛硬币一样，我们可以把每个神经元当做抛硬币一样
    #硬币 正面的概率为p，n表示每个神经元试验的次数
    #因为我们每个神经元只需要抛一次就可以了所以n=1，size参数是我们有多少个硬币。
    sample=np.random.binomial(n=1,p=retain_prob,size=x.shape)#即将生成一个0、1分布的向量，0表示这个神经元被屏蔽，不工作了，也就是dropout了
    print('sample:',sample)
    x *=sample#0、1与x相乘，我们就可以屏蔽某些神经元，让它们的值变为0
    print('result:',x)
    x /= retain_prob
 
    return x
#对dropout的测试，大家可以跑一下上面的函数，了解一个输入x向量，经过dropout的结果
for i in range(0,10):
    x=np.asarray([1,2,3,4,5,6,7,8,9,10],dtype=np.float32)
    dropout(x,0.4)

```
结果，每一次都不一样，其中一个可能的样例：
```python
sample: [0 1 1 1 0 1 0 1 1 0]
result: [0. 2. 3. 4. 0. 6. 0. 8. 9. 0.]
sample: [1 1 0 0 0 0 0 1 1 0]
result: [1. 2. 0. 0. 0. 0. 0. 8. 9. 0.]
sample: [1 0 1 1 0 0 1 0 0 1]
result: [ 1.  0.  3.  4.  0.  0.  7.  0.  0. 10.]
sample: [1 0 0 0 0 0 0 0 0 0]
result: [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sample: [1 1 0 0 1 0 1 1 0 0]
result: [1. 2. 0. 0. 5. 0. 7. 8. 0. 0.]
sample: [1 0 0 0 0 0 1 1 1 1]
result: [ 1.  0.  0.  0.  0.  0.  7.  8.  9. 10.]
sample: [1 1 1 0 0 1 0 0 0 1]
result: [ 1.  2.  3.  0.  0.  6.  0.  0.  0. 10.]
sample: [1 1 1 0 1 0 1 0 1 0]
result: [1. 2. 3. 0. 5. 0. 7. 0. 9. 0.]
sample: [1 1 1 1 0 1 0 1 1 1]
result: [ 1.  2.  3.  4.  0.  6.  0.  8.  9. 10.]
sample: [1 1 0 1 1 0 1 0 0 0]
result: [1. 2. 0. 4. 5. 0. 7. 0. 0. 0.]


```






#### Initialize_weights

初始化参数是一个非常重要的事情
我们在之前的复现中实现了一个初始化参数的方法，对其修改后我们看看这个初始化方法是如何工作的：

```python

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

```
可以看到在遍历整个module的过程中
显示判断了遍历到的对象属于卷积，BN层还是线性层

一共出现了三种可自定义初始化的参数也就是
  

```python
BN:
 tensor([0.7880, 0.7868, 0.4304, 0.7028, 0.7943, 0.9234, 0.9547, 0.6654, 0.1503,
        0.3750, 0.4263, 0.6845, 0.6919, 0.7447, 0.5500, 0.7412, 0.3376, 0.9028,
        0.3876, 0.6891, 0.1153, 0.7431, 0.7058, 0.8124, 0.9490, 0.3707, 0.2683,
        0.8208, 0.4948, 0.1184, 0.3112, 0.3360, 0.0200, 0.0354, 0.7110, 0.7964,
        0.5461, 0.9730, 0.1676, 0.4202, 0.3160, 0.5562, 0.3004, 0.4339, 0.6145,
        0.7794, 0.0471, 0.5734, 0.1839, 0.0627, 0.8915, 0.4891, 0.5451, 0.6070,
        0.1087, 0.4008, 0.7421, 0.1131, 0.8380, 0.9787, 0.2596, 0.8935, 0.9562,
        0.2018])
Modified BN:
 tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])


Linear:
 tensor([[ 0.0129, -0.0137, -0.0083,  ...,  0.0108,  0.0107, -0.0068],
        [ 0.0156,  0.0119, -0.0045,  ...,  0.0082, -0.0042, -0.0036],
        [-0.0155, -0.0015, -0.0037,  ...,  0.0094, -0.0005, -0.0145],
        ...,
        [ 0.0042,  0.0047,  0.0149,  ..., -0.0068,  0.0063,  0.0131],
        [-0.0089,  0.0107, -0.0097,  ...,  0.0080, -0.0080,  0.0154],
        [-0.0019,  0.0116, -0.0035,  ...,  0.0156, -0.0054,  0.0047]])
Modified Linear:
 tensor([[-0.0203, -0.0030,  0.0009,  ..., -0.0016,  0.0027,  0.0022],
        [-0.0263,  0.0071,  0.0217,  ...,  0.0089,  0.0080,  0.0011],
        [-0.0052, -0.0160, -0.0018,  ...,  0.0181, -0.0130, -0.0068],
        ...,
        [ 0.0155, -0.0091, -0.0032,  ...,  0.0013, -0.0043, -0.0198],
        [-0.0027,  0.0061,  0.0277,  ...,  0.0073,  0.0285, -0.0009],
        [ 0.0085, -0.0064, -0.0046,  ...,  0.0161, -0.0018,  0.0024]])
ReLU(inplace) 


```

#### isinstance

描述
isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。

 * isinstance() 与 type() 区别：

type() 不会认为子类是一种父类类型，不考虑继承关系。

isinstance() 会认为子类是一种父类类型，考虑继承关系。

如果要判断两个类型是否相同推荐使用 isinstance()。

语法
以下是 isinstance() 方法的语法:

isinstance(object, classinfo)
参数
object -- 实例对象。
classinfo -- 可以是直接或间接类名、基本类型或者由它们组成的元组。
```python
class A:
    pass
 
class B(A):
    pass
 
isinstance(A(), A)    # returns True
type(A()) == A        # returns True
isinstance(B(), A)    # returns True
type(B()) == A        # returns False
```


#### forward

#### tensor.view
把原先tensor中的数据按照行优先的顺序排成一个一维的数据（这里应该是因为要求地址是连续存储的），然后按照参数组合成其他维度的tensor。比如说是不管你原先的数据是[[[1,2,3],[4,5,6]]]还是[1,2,3,4,5,6]，因为它们排成一维向量都是6个元素，所以只要view后面的参数一致，得到的结果都是一样的。

```python
a=torch.Tensor([[[1,2,3],[4,5,6]]])
b=torch.Tensor([1,2,3,4,5,6])

print(a.view(1,6))
print(b.view(1,6))
```

结果都是tensor([[1., 2., 3., 4., 5., 6.]]) 
所有向量都合并为一维向量



#### transfrom

数据的变换


#### 





## 卷积操作与卷基层
卷及操作本质上



## 池化操作与池化层




## Linear



## BasicBlock & Bottleneck


## Softmax


![](https://pic4.zhimg.com/80/v2-11758fbc2fc5bbbc60106926625b3a4f_hd.jpg)


## Loss Cross Entropy





## Train & Test



## ResNet

ResNet相对于VGG等传统的朴素CNN来说最大的区别就是加入了short-cut,其基础结构如下:



![](https://img-blog.csdn.net/20171223111002643?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvSmluZ194aWFu/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)











# WS-Faster-RCNN
