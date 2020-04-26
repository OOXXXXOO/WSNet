import numpy as np
import pandas as pd
# class IO():
#     def __init__(self,Unit,size):
#         self.Unit=Unit
#         self.size=size

import matplotlib.pyplot as plt      
a=np.load("/workspace/data/38cloud-cloud-segmentation-in-satellite-images/cloud1percent5000.npy",allow_pickle=True)
image=a[0]["imagery"]
print(image.shape)
plt.imshow(image),plt.show()