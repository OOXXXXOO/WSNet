import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
import torch
import numpy as np
import torchvision
img='DocResources/quick.jpg'
imgg=Image.open(img)
print(imgg.size)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)
print(predictions)