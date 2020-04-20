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


# STF-Smart Transform Tool
The traditional transform process need different function ,class, parameter and random process. The smart transform will don't need set any parameter to make data augmentation with smart recognize type of mission and data.

As first of all , we need rule the format of data that input to transform . 

input `__call__` function expected two para:  
* image
`PIL image or numpy ndarray` 
* target
```python
    dict={
        
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W
        
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance
    }
```
**Mode:**
* Detection
        target dict:             
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label for each ground-truth box
* InstanceSegmentation
        target dict :
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
        between 0 and H and 0 and W
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance
* Segmentation
        target dict :
        Output [(Batch_Size),W,H,CLASS_NUM] argmax(Axis=1) with w*h*c => [(Batch_Size),W,H]
        Target [(Batch_Size),W,H] value is classes index