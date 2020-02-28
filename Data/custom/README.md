## This Custom Dataset Doecument


* Satellite imagery dataset generate
The satellite imagery has huge pixel count like $10000^2$ pixels
so the single imagery can't be train & inference directly. we should random resample the imagery to build dataset.
In the [/Src/Utils/DataToolkit/labelme2coco.py](./../../Src/Utils/DataToolkit/README.md) we could transfom&visualize  the labelme label dataset 



like:
```bash
|---root
    |---raw_data
        |---1.tif
        |---2.tif
    |---dataset
        |---labels(optional for segmentation)
            |---label1.png(or npy,npz)
            |---label2.png...
        |---viz
            |---label1_viz.png
            |---label2_viz.png
        |---annotations.json(for detection & instance segmentation)
        |---labels.txt
```


The `custom.py`could  **Random - Resample** the list of whole imagery and label to generate the dataset like:


##### Auto Index - Seperated Segmentation Dataset 
```bash
|---images
    |---000001.png
    |---000002.png
    |---...
|---labels
    |---000001.png
    |---000001.png
    |---...
|---viz
    |---000001.png
    |---000002.png
    |---...
|---labels.txt

```
##### Auto Index - Seperted Detection / InstanceSegmentation / Keypoint / Caption Dataset (COCO like)
```bash
|---root
    |---dataset
        |---images
            |---000001.png
            |---000002.png
        |---viz
            |---label1_viz.png
            |---label2_viz.png
        |---annotations.json
```
##### Index Include - Fast dataset for Segmentation
```JSON
dataset.npy/npz
np.ndarray
[
    {
        "__backgrond__":0,
        "class1":1,}
    {
        "imagedata":np.ndarray
        "label":np.ndarray
    },
    .....
]
```



* custom data format don't need resample
[see here](./../../Src/Utils/DataToolkit/README.md)
