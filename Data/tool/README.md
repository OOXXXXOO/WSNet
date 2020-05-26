# Imagery Toolkit:
## 1.Temdet
Project use thg NMS＆Template that choosen by manual work algorithm to detect object with minimum interclass variance.

<!--
 Copyright 2020 winshare
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->


### Algorithm Guide



```python
import matplotlib.pyplot as plt
from temdet import nmsdet
detector=nmsdet("/workspace/data/clip/clip1.tif",CHANNEL_INDEX=[0,1,2,3])#For NDVI
#detector=nmsdet("/workspace/data/clip/clip1.tif",CHANNEL_INDEX=[0,1,2])#For RGB

```

    # ---------------------------------------------------------------------------- #
    #                    NMS&Template MApping Toolkit                              #
    # ---------------------------------------------------------------------------- #
    -----Template NMS detect in : 
     /workspace/data/clip/clip1.tif 
     Mon Apr 27 16:21:57 2020
    # ---------------------------------------------------------------------------- #
    #                            TIFF process Toolkit                              #
    # ---------------------------------------------------------------------------- #
    -----TIFF Class Init with : /workspace/data/clip/clip1.tif
    -----Original Data Shape :  (4, 1303, 1631)
    image type : int16
    image shape (1303, 1631, 4)



```python
ndvi=detector.buildndvi()
plt.imshow(ndvi),plt.show()
```




```python
detector.addbox(0,0,100,100)
detector.detect(drawbox=True)
print("boxes",detector.boxes,"conf",detector.conf)
```



    boxes [(0, 0, 100, 100), (100, 100, 200, 200)] conf [0.78325445 0.79081626]



