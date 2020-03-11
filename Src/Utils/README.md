## Utils























## Data Bridge:

神经网络面对多任务最大的麻烦就是不同网络组件之间的数据结构统一，因此本部分总结了常规Pytorch官方

对于非图像标签的数据而言  Pytorch官方实现主要面对的是Instance Segmentation, KeyPoint ,Detection,Caption四种任务类型,所有的标签以数据字典的形式存储与IO.
分别对应的是： 

####　MaskRCNN
```
* boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

* labels (Int64Tensor[N]): the predicted labels for each image

* scores (Tensor[N]): the scores or each prediction

* masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range.
```

#### FasterRCNN
```
* boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

* labels (Int64Tensor[N]): the class label for each ground-truth box
```
#### KeyPointRCNN
```
* boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

* labels (Int64Tensor[N]): the class label for each ground-truth box

* keypoints (FloatTensor[N, K, 3]): the K keypoints location for each of the N instances, in the format [x, y, visibility], where visibility=0 means that the keypoint is not visible.
```
值得注意的是，对数据字典的解析中官方实现中自带了Transform，因此必须要考虑这个问题。
****


对于图像标签数据而言，官方实现的网络限制较小,只要求了面对输入数据的normalize参数

#### Segmentation

The images have to be loaded in to a range of [0, 1] and then normalized using：
```
* mean = [0.485, 0.456, 0.406] 
* std = [0.229, 0.224, 0.225]. 
```

结合以上我们可以得出总的标签桥接数据标准只有两类：
* 数据字典
* 图像标签

在网络输入，输出，训练,推理时,各个数据的桥接，转换，判断就十分重要