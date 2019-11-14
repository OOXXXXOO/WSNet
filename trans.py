import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
img='DocResources/quick.jpg'
imgg=Image.open(img)
imggg=T.functional.pad(imgg,10)
imggg.show()