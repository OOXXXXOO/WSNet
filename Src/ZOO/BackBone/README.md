![](./../../../Document/Image/title.png)





## Zoo.BackBone


### Accuracy on validation set (single model)

Results were obtained using (center cropped) images of the same size than during the training process.

Model | Version | Acc@1 | Acc@5
--- | --- | --- | ---
PNASNet-5-Large | [Tensorflow](https://github.com/tensorflow/models/tree/master/research/slim) | 82.858 | 96.182
[PNASNet-5-Large](https://github.com/Cadene/pretrained-models.pytorch#pnasnet) | Our porting | 82.736 | 95.992
NASNet-A-Large | [Tensorflow](https://github.com/tensorflow/models/tree/master/research/slim) | 82.693 | 96.163
[NASNet-A-Large](https://github.com/Cadene/pretrained-models.pytorch#nasnet) | Our porting | 82.566 | 96.086
SENet154 | [Caffe](https://github.com/hujie-frank/SENet) | 81.32 | 95.53
[SENet154](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting | 81.304 | 95.498
PolyNet | [Caffe](https://github.com/CUHK-MMLAB/polynet) | 81.29 | 95.75
[PolyNet](https://github.com/Cadene/pretrained-models.pytorch#polynet) | Our porting | 81.002 | 95.624
InceptionResNetV2 | [Tensorflow](https://github.com/tensorflow/models/tree/master/slim) | 80.4 | 95.3
InceptionV4 | [Tensorflow](https://github.com/tensorflow/models/tree/master/slim) | 80.2 | 95.3
[SE-ResNeXt101_32x4d](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting | 80.236 | 95.028
SE-ResNeXt101_32x4d | [Caffe](https://github.com/hujie-frank/SENet) | 80.19 | 95.04
[InceptionResNetV2](https://github.com/Cadene/pretrained-models.pytorch#inception) | Our porting | 80.170 | 95.234
[InceptionV4](https://github.com/Cadene/pretrained-models.pytorch#inception) | Our porting | 80.062 | 94.926
[DualPathNet107_5k](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting | 79.746 | 94.684
ResNeXt101_64x4d | [Torch7](https://github.com/facebookresearch/ResNeXt) | 79.6 | 94.7
[DualPathNet131](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting | 79.432 | 94.574
[DualPathNet92_5k](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting | 79.400 | 94.620
[DualPathNet98](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting | 79.224 | 94.488
[SE-ResNeXt50_32x4d](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting | 79.076 | 94.434
SE-ResNeXt50_32x4d | [Caffe](https://github.com/hujie-frank/SENet) | 79.03 | 94.46
[Xception](https://github.com/Cadene/pretrained-models.pytorch#xception) | [Keras](https://github.com/keras-team/keras/blob/master/keras/applications/xception.py) | 79.000 | 94.500
[ResNeXt101_64x4d](https://github.com/Cadene/pretrained-models.pytorch#resnext) | Our porting | 78.956 | 94.252
[Xception](https://github.com/Cadene/pretrained-models.pytorch#xception) | Our porting | 78.888 | 94.292
ResNeXt101_32x4d | [Torch7](https://github.com/facebookresearch/ResNeXt) | 78.8 | 94.4
SE-ResNet152 | [Caffe](https://github.com/hujie-frank/SENet) | 78.66 | 94.46
[SE-ResNet152](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting | 78.658 | 94.374
ResNet152 | [Pytorch](https://github.com/pytorch/vision#models) | 78.428 | 94.110
[SE-ResNet101](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting | 78.396 | 94.258
SE-ResNet101 | [Caffe](https://github.com/hujie-frank/SENet) | 78.25 | 94.28
[ResNeXt101_32x4d](https://github.com/Cadene/pretrained-models.pytorch#resnext) | Our porting | 78.188 | 93.886
FBResNet152 | [Torch7](https://github.com/facebook/fb.resnet.torch) | 77.84 | 93.84
SE-ResNet50 | [Caffe](https://github.com/hujie-frank/SENet) | 77.63 | 93.64
[SE-ResNet50](https://github.com/Cadene/pretrained-models.pytorch#senet) | Our porting | 77.636 | 93.752
[DenseNet161](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 77.560 | 93.798
[ResNet101](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 77.438 | 93.672
[FBResNet152](https://github.com/Cadene/pretrained-models.pytorch#facebook-resnet) | Our porting | 77.386 | 93.594
[InceptionV3](https://github.com/Cadene/pretrained-models.pytorch#inception) | [Pytorch](https://github.com/pytorch/vision#models) | 77.294 | 93.454
[DenseNet201](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 77.152 | 93.548
[DualPathNet68b_5k](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting | 77.034 | 93.590
[CaffeResnet101](https://github.com/Cadene/pretrained-models.pytorch#caffe-resnet) | [Caffe](https://github.com/KaimingHe/deep-residual-networks) | 76.400 | 92.900
[CaffeResnet101](https://github.com/Cadene/pretrained-models.pytorch#caffe-resnet) | Our porting | 76.200 | 92.766
[DenseNet169](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 76.026 | 92.992
[ResNet50](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 76.002 | 92.980
[DualPathNet68](https://github.com/Cadene/pretrained-models.pytorch#dualpathnetworks) | Our porting | 75.868 | 92.774
[DenseNet121](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 74.646 | 92.136
[VGG19_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 74.266 | 92.066
NASNet-A-Mobile | [Tensorflow](https://github.com/tensorflow/models/tree/master/research/slim) | 74.0 | 91.6
[NASNet-A-Mobile](https://github.com/veronikayurchuk/pretrained-models.pytorch/blob/master/pretrainedmodels/models/nasnet_mobile.py) | Our porting | 74.080 | 91.740
[ResNet34](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 73.554 | 91.456
[BNInception](https://github.com/Cadene/pretrained-models.pytorch#bninception) | Our porting | 73.524 | 91.562
[VGG16_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 73.518 | 91.608
[VGG19](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 72.080 | 90.822
[VGG16](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 71.636 | 90.354
[VGG13_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 71.508 | 90.494
[VGG11_BN](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 70.452 | 89.818
[ResNet18](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 70.142 | 89.274
[VGG13](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 69.662 | 89.264
[VGG11](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 68.970 | 88.746
[SqueezeNet1_1](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 58.250 | 80.800
[SqueezeNet1_0](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 58.108 | 80.428
[Alexnet](https://github.com/Cadene/pretrained-models.pytorch#torchvision) | [Pytorch](https://github.com/pytorch/vision#models) | 56.432 | 79.194


## SOTA Model

### EfficientNet
Note that pretrained models have only been released for `N=0,1,2,3,4,5` at the current time, so `.from_pretrained` only supports `'efficientnet-b{N}'` for `N=0,1,2,3,4,5`. 

Details about the models are below: 

|    *Name*         |*# Params*|*Top-1 Acc.*|*Pretrained?*|
|:-----------------:|:--------:|:----------:|:-----------:|
| `efficientnet-b0` |   5.3M   |    76.3    |      ✓      |
| `efficientnet-b1` |   7.8M   |    78.8    |      ✓      |
| `efficientnet-b2` |   9.2M   |    79.8    |      ✓      |
| `efficientnet-b3` |    12M   |    81.1    |      ✓      |
| `efficientnet-b4` |    19M   |    82.6    |      ✓      |
| `efficientnet-b5` |    30M   |    83.3    |      ✓      |
| `efficientnet-b6` |    43M   |    84.0    |      ✓      |
| `efficientnet-b7` |    66M   |    84.4    |      ✓      |
