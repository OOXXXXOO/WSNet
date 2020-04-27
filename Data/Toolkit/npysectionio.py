import numpy as np
import pandas as pd
# class IO():
#     def __init__(self,Unit,size):
#         self.Unit=Unit
#         self.size=size

import matplotlib.pyplot as plt      
a=np.load("/workspace/data/38cloud-cloud-segmentation-in-satellite-images/ProcessedDataset/cloud1percent5000.npy",allow_pickle=True)
image=a[0]["imagery"]
target=a[0]["target"]
print(image.shape)
plt.imshow(target),plt.show()