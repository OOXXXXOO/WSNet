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
![](Resources/Document/IMG_0932.PNG)


**WSNets is beta version fast deeplearning training framework base on pytorch 1.3.**


This project work for train, validation, test, with general dataset by different model or use the custom dataset that generate by specified data structure.


The major purpose of this project :

* Start a fast training instance on the standard datasets with default pre-trained model.
*  Automatic build the dataset from original images & labels with **the specified data structure rule**.
* Support for some state-of-the-art models.  
* Use Json config file configure the training / validation /inference instance to avoid change the project code.

****

#### Main Features:

##### 1.DataSet Generator

*  Automatic build the dataset from original images & labels with **the specified data structure rule**. In addition, it use the strict data format to help people fix the format error.


##### 2.Fast traing from template config file

* Use Json config file configure the training / validation /inference instance to avoid change the project code.



##### 3.Instance like training management 

* Start a fast training instance on the standard or generated datasets with default pre-trained model.


##### 4.Great automatic process & visulization 


* Config decode will show up on terminal & process info will add to tensorboard where you could see the perference of training processing. 


#### The important toolkit 


##### Utils
* [STF (Smart transform module)](Src/Utils/Transform/README.md)
* [Neural Network Analysis module](Src/Utils/Transform/README.md)
* [Evaluator](Src/Utils/Evaluator/README.md)

##### Neural Network Module
* [PointRend](Src/Nets/Module/PointRend/README.md)
* [DensePose](Src/Nets/Module/DensePose/README.md)
* [CSRC { BN,DC,mask,nms,roi_align,rotated_box,wrappers }](Src/Nets/Layers/csrc/README.md)

##### DataSet

* [temdet](Data/Toolkit/temdet/README.md)
* [labelme2coco](Data/README.md)
* [patch generate](Data/README.md)
* [SatelliteData](Data/Toolkit/satellite_imagery/README.md)

### Project Design

[Description](Resources/Document/DesignDescription.md)



> [Update schedule](UpdatePlanning.md) 


### Document / 文档:

#### EN:

****


| [Installation](Resources/Document/Installation.md)| [Guide to Start](./Resources/Document/Guide2start.md) | [DataSet Toolkit](./Src/Utils/DataToolkit/README.md) | [NetWork](./Src/Nets/README.md) | [DataSet](./Data/README.md)| [Transform](./Src/Utils/Transform/README.md) | [Config](./Config/README.md) |
|---|---|---|---|---|---|---|

****


#### 中文:
****
| [安装](Resources/Document/Installation.md) | [快速开始](./Resources/Document/Guide2start.md)| [数据集工具](./Src/Utils/DataToolkit/README.md) | [神经网络](./Src/Nets/README.md) | [数据集](./Data/README.md) | [数据变换](./Src/Utils/Transform/README.md) | [配置](./Config/README.md) |
|---|---|---|---|---|---|---|

****

**Support Mission Table:**

|   | Segmentation  | Detection  | Instance Segmentation |
|---|---|---|---|
| COCO  | √  | √  |  √ |
|  NPY |  √ |  √ |  √ |
|  CitysCapes | √  |   |   |
| Pascal VOC  |   |  √ |   |
| CustomDataSet  |  √ | √  | √  |
| Open Image  |   | √  |   |




### Copyright 2020 winshare
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.