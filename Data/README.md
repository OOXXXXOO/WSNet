# DataSet


This framework recommend COCO format to do the Detection, Instance Segmentation,and Segmentation Mission training. So we have the  toolkit to translate and generate COCO format dataset.

[DataSet Toolkit Document](Data/datatoolkit/README.md)

As we all know the official COCO is a json-base annotation format standard that could support for different mission , the modified COCO annotation format unit have more **info key value** like : 

* **file_name**: the full path to the image file. 
  
* **sem_seg_file_name**: the full path to the ground truth semantic segmentation file.


* **sem_seg**: semantic segmentation ground truth in a `2D torch.Tensor`. Values in the array represent `category labels starting from 0`.

* **height, width**: `integer`. The shape of image.

* **image_id** `(str or int)`: a unique id that identifies this image. Used during evaluation to identify the images, but a dataset may use it for different purposes.

* **annotations** `(list[dict])`: each dict corresponds to annotations of one instance in this image. Images with empty annotations will by default be removed from training,

* **bbox** `(list[float])`: list of 4 numbers representing the bounding box of the instance.

* **bbox_mode** `(int)`: the format of bbox. It must be a member of structures.BoxMode. Currently supports: BoxMode.XYXY_ABS, BoxMode.XYWH_ABS.

* **category_id** `(int)`: an integer in the range [0, num_categories) representing the category label. The value num_categories is reserved to represent the “background” category, if applicable.

* **segmentation** `(list[list[float]] or dict)`:If `list[list[float]]`, it represents a list of polygons, one for each connected component of the object. Each `list[float]` is one simple polygon in the format of `[x1, y1, ..., xn, yn]`.
    The `Xs` and `Ys` are either relative coordinates in [0, 1], or absolute coordinates, depend on whether `“bbox_mode”` is relative.

    If dict, it represents the per-pixel segmentation mask in COCO’s RLE format. The dict should have keys “size” and “counts”. 

    You can convert a uint8 segmentation mask of `0s` and `1s` into RLE format by `pycocotools.mask.encode(np.asarray(mask, order="F"))`.

* **keypoints** `(list[float])`: in the format of `[x1, y1, v1,…, xn, yn, vn]`. `v[i]` means the visibility of this keypoint. n must be equal to the number of keypoint categories. The Xs and Ys are either relative coordinates in [0, 1], or absolute coordinates, depend on whether “bbox_mode” is relative.

    Note that the coordinate annotations in COCO format are integers in range `[0, H-1 or W-1]`. By default, detectron2 adds 0.5 to absolute keypoint coordinates to convert them from discrete pixel indices to floating point coordinates.

* **iscrowd**: `0 or 1`. Whether this instance is labeled as COCO’s “crowd region”. **Don’t include this field if you don’t know what it means.**
  
we could demonstrate a demo json:

```json
{
    "images": [
        {
            "height": 768,
            "width": 1024,
            "id": 1,
            "file_name": "hourse.jpg"
        }
    ],
    "categories": [
        {
            "supercategory": "h",
            "id": 2,
            "name": "o"
        },
       
    ],
    "annotations": [
        {
            "segmentation": [
                [
                    92.0392156862745,
                    708.2352941176471,
                    91.6470588235294,
                    689.4117647058824,
                    93.6078431372549,
                    667.8431372549019,
                    94.3921568627451,
                    646.6666666666667,
                    99.49019607843137,
                    625.4901960784314,
                    109.29411764705883,
                    592.5490196078431,
                    114.0,
                    560.7843137254902,
                    113.21568627450979,
                    532.1568627450981,
                ]
            ],
            "iscrowd": 0,
            "image_id": 1,
            "bbox": [
                91.0,
                26.0,
                770.0,
                741.0
            ],
            "category_id": 2,
            "id": 1
        },
        ......
    ]
}
```

Support Pytorch Official Dataset Function :

```python
        self.datasets_function_dict={
            "Classification":{
                "MINST":dataset.MNIST,
                "FashionMINST":dataset.FashionMNIST,
                "KMINST":dataset.KMNIST,
                "EMINST":dataset.EMNIST,
                "CIFAR10":dataset.CIFAR10,
                "CIFAR100":dataset.CIFAR100,
                "ImageNet":dataset.ImageNet
            },
            "Detection":{
                "CocoDetection":dataset.CocoDetection,
                "VOC_Detection":dataset.VOCDetection
            },
            "Segmentation":{
                "VOC_Segmentation":dataset.VOCSegmentation,
                "Cityscapes":dataset.Cityscapes,
                "CocoDetection":dataset.CocoDetection,
                "Costum_NPY_DataSet":Costum_NPY_DataSet
            },
            "Caption":{
                "CocoCaptions":dataset.CocoCaptions
            },
            "InstanceSegmentation":{
                "CocoDetection":dataset.CocoDetection # Need Modified to get mask
            }
        }
```


## Custom DataSet

## DataSet ToolKit


## COCO Detection


## COCO Instance Sgementation


## CistysCapes


## Pascal VOC

中文说明:

**文件名**：图像文件的完整路径。

* **sem_seg_file_name**：语义分割标签文件的完整路径。



* **sem_seg**：2D `torch.tensor`中的语义分割标签mask。数组中的值代表“从0开始的类别标签”。

* ***height，width**：`integer`。图像的长宽。

* **image_id**`（str或int）`：标识此图像的唯一ID。在评估期间用于识别图像，但数据集可将其用于不同目的。

* **annotations**`（list [dict]）`：每个字典对应此图像中一个实例的注释。默认情况下，带有空注释的图像将被从训练中删除，但可以使用DATALOADER.FILTER_EMPTY_ANNOTATIONS将其包括在内。每个字典可能包含以下键：

* **bbox**`（list [float]）`：代表实例边界框的4个数字的列表。

* **bbox_mode**`（int）`：bbox的格式。它必须是structure.BoxMode的成员。当前支持：BoxMode.XYXY_ABS，BoxMode.XYWH_ABS。

* **category_id**`（int）`：表示类别标签的[0，num_categories）范围内的整数。保留值num_categories以表示“背景”类别（如果适用）。

* **segmentation**`（（list [list [float]]或dict）]`：如果list [list [float]]表示多边形的列表，则该对象的每个连接组件均包含一个多边形。每个`list [float]`是一个简单的多边形，格式为[[x1，y1，...，xn，yn]`
    Xs和Ys是[0，1]中的相对坐标，还是绝对坐标，取决于“ bbox_mode”是相对的。
    如果是dict，则以COCO的RLE格式表示每个像素的分割蒙版。字典应具有键“大小”和“计数”。

    您可以通过`pycocotools.mask.encode（np.asarray（mask，order =“ F”））`将uint8分段掩码0s和1s转换为RLE格式。

* **keypoints**`（list [float]）`：格式为[[x1，y1，v1，…，xn，yn，vn]`。 v[i]表示此关键点的可见性。n必须等于关键点类别的数量。 Xs和Ys是[0，1]中的相对坐标，还是绝对坐标，取决于“ bbox_mode”是否是相对的。

    注意，COCO格式的坐标注释是范围`[[0，H-1或W-1]`的整数。默认情况下，detectron2向绝对关键点坐标添加0.5，以将其从离散像素索引转换为浮点坐标。

* **iscrowd**：`0或1`。此实例是否被标记为COCO的“人群区域”。如果您不知道这是什么意思，请不要包含此字段。 
