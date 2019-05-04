import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import measure
imagedir='/home/winshare/下载/1723236901.jpg'
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)
import numpy as np
origin=cv2.imread(imagedir)
gray=rgb2gray(origin)
print(gray.mean()*255)
mean1=gray.mean()
gray[gray>mean1]=0
image = inverse_gaussian_gradient(gray, alpha=20, sigma=0.9)
image=1-image
# gray1=np.float(gray)
contours=measure.find_contours(origin[:,:,0],mean1*255)
print(contours)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10),
                            sharex=True, sharey=True)

ax[0].imshow(origin,cmap=plt.cm.nipy_spectral, interpolation='nearest')
ax[1].imshow(origin)
for n, contour in enumerate(contours):
    ax[1].plot(contour[:, 1], contour[:, 0], linewidth=2)
plt.show()

print(image.shape)