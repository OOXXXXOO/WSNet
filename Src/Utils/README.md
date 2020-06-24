
# Utils



### CNN Summary toolkit & GRam 

#### Usage:
```python
from Src.Utils.Analysis import summary,graminfo
import torchvision.models as models
model= models.resnet101(pretrained=False)
model.eval()
_,total_size=summary(model,(3,512,512),batch_size=4,device="cuda:0")
graminfo()
```
#### Output
```bash
# ---------------------------------------------------------------------------- #
#                      summary module for backbone network                     #
# ---------------------------------------------------------------------------- #
# ============================================================================ #
        Layer (type)               Output Shape           Param
# ============================================================================ #
#                      Conv2d-1          [4, 64, 256, 256]           9,408     #
#                 BatchNorm2d-2          [4, 64, 256, 256]             128     #
#                        ReLU-3          [4, 64, 256, 256]               0     #
#                   MaxPool2d-4          [4, 64, 128, 128]               0     #
#                      Conv2d-5          [4, 64, 128, 128]           4,096     #
.
.
NetworkLayerinfo
.
.
#                Bottleneck-342          [4, 2048, 16, 16]               0     #
#         AdaptiveAvgPool2d-343            [4, 2048, 1, 1]               0     #
#                    Linear-344                  [4, 1000]       2,049,000     #
#                    ResNet-345                  [4, 1000]               0     #
# ============================================================================ #
# Total params: 44,549,160
# Trainable params: 44,549,160
# Non-trainable params: 0
# ============================================================================ #
# Input size (Pixel): [(3, 512, 512)]
# Input size (MB): 12.00
# Forward/backward pass size (MB): 8980.12
# Params size (MB): 169.94
# Estimated Total Size (MB): 9162.06
# ============================================================================ #
# ============================================================================ #
# ===== Used GRAM:2.531982421875 GB - 10.75811767578125
# ===== Free GRAM:8.22613525390625 GB - 10.75811767578125
# ============================================================================ #
```