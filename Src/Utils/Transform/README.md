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


### The General Transform 

The traditional transform just has image transform,but target tansform is must modified by dataset object that make Random transform couldn't easy to use. 

The official dataset function don't have anyway to make transform support for all of them. So the General Transform class is a resolution :
it can generate three way of transforms with one config file

* transforms(image,target)
* targets_transforms(target)
* images_transforms(images)



Usage:
            
transform: list of transform dict in config json file like:

```json
[
    {"RandomSizedCrop":512},
    {"RandomRotation":90},
    {"Normalize":[[0.485,0.456,0.406],[0.229, 0.224, 0.225]]}
    {"ToTensor":"None"}
]
```



**Member:**
```python
self.target_transform : list of target transform
self.image_transform: list of image transform
```

```
        Random transform will be different  
        
        Whole Structurelike:
                                        |->ndarray_transform<---------|______
                             |->target->|->dict_vector_transform <----|filter|
        keys->transformlist->|                                        |
                             |->image-->|->image_transform->|->random_para_dict

        The target in different mission just include two support way:
        image->image mask label | ndarray
        image->dict label       | dict
```
Just like you see, The target will be classified to dict or ndarray.


    
the **dict target point sets transform** support:
```json
    "Pad":T.Pad,
    "RandomCrop":T.RandomCrop,
    "RandomErasing":T.RandomErasing,
    "RandomHorizontalFlip":T.RandomHorizontalFlip,
    "RandomPerspective":T.RandomPerspective,
    "RandomResizedCrop":T.RandomResizedCrop,
    "RandomRotation":T.RandomRotation,
    "RandomSizedCrop":T.RandomSizedCrop,
    "RandomVerticalFlip":T.RandomVerticalFlip,
    "Resize":T.Resize,
    "Scale":T.Scale,
    "TenCrop":T.TenCrop
```

* if the target is dictionry:
  
    all the position relate transform will be process on point sets in annotation .

the **ndarray target transform** support :
```json
    "Pad":T.Pad,
    "RandomAffine":T.RandomAffine,
    "RandomApply":T.RandomApply,
    "RandomChoice":T.RandomChoice,
    "RandomCrop":T.RandomCrop,
    "RandomErasing":T.RandomErasing,
    "RandomHorizontalFlip":T.RandomHorizontalFlip,
    "RandomOrder":T.RandomOrder,
    "RandomPerspective":T.RandomPerspective,
    "RandomResizedCrop":T.RandomResizedCrop,
    "RandomRotation":T.RandomRotation,
    "RandomSizedCrop":T.RandomSizedCrop,
    "RandomVerticalFlip":T.RandomVerticalFlip,
    "Resize":T.Resize,
    "Scale":T.Scale,
    "TenCrop":T.TenCrop,
    "ToPILImage":T.ToPILImage,
    "ToTensor":T.ToTensor

```
* if the target is ndarray:
  
    will process all transform after remove the pixel value relate transforms.

For easy to understand , we recommend use the minimal set to avoid compatibility problem




#### Full Support Dict:
```json
{
    "Grayscale":T.Grayscale,
    "Lambda":T.Lambda,
    "Normalize":T.Normalize,
    "Pad":T.Pad,
    "RandomAffine":T.RandomAffine,
    "RandomApply":T.RandomApply,
    "RandomChoice":T.RandomChoice,
    "RandomCrop":T.RandomCrop,
    "RandomErasing":T.RandomErasing,
    "RandomGrayscale":T.RandomGrayscale,
    "RandomHorizontalFlip":T.RandomHorizontalFlip,
    "RandomOrder":T.RandomOrder,
    "RandomPerspective":T.RandomPerspective,
    "RandomResizedCrop":T.RandomResizedCrop,
    "RandomRotation":T.RandomRotation,
    "RandomSizedCrop":T.RandomSizedCrop,
    "RandomVerticalFlip":T.RandomVerticalFlip,
    "Resize":T.Resize,
    "Scale":T.Scale,
    "TenCrop":T.TenCrop,
    "ToPILImage":T.ToPILImage,
    "ToTensor":T.ToTensor,
}
```