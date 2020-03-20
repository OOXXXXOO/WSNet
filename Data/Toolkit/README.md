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





## Patched Dataset Generator:















## DataToolkit Document


### labelmejson transform to coco annotation

* This script work for transform a set of annotated `Labelme JSON` files to one COCO201x-like JSON file. 
* Support for Detection,Semantic Segmantation ,Instance Segmentation. 

### usage:
    
    labelme2coco.py [-h] [--a A] [--o O] [--v V] [--m M]

#### optional arguments:


    optional arguments:
    
    -h, --help  show this help message and exit
    --a A       dir of anno file like ./annotation/
    --o O       dir of output annotation file like annotation.json
    --v V       bool type about output label visualization or not
    --m M       box format mode like support : XYWH_ABS(Default for COCO), XYXY_ABS



#### example:

```bash
python labelme2coco.py  --a ./labelme/demo/train2014/ --o ./dataset --v True
```

The Script will be generate :
* `labels.txt` include the class name of whole cocojson file
* The `annotation.json` with **COCO JSON** format,
* The Label file of Mask
* The Visualization of label:
![](./dataset/viz/cat-11_viz_label.png)

****

The transformed folder structure like:

```bash
---root
    |---dataset
        |---label
            |---label1.png(or npy,npz)
            |---label2.png...
        |---viz
            |---label1_viz.png
            |---label2_viz.png
        |---annotations.json
```