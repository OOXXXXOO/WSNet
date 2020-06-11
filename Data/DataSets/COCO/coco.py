# Copyright 2020 winshare
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    coco.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/03/20 16:14:51 by winshare          #+#    #+#              #
#    Updated: 2020/03/20 16:14:51 by winshare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


from PIL import Image
import os
import os.path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
import numpy as np



class CocoDataset(COCO):
    """
    MS Coco Support <http://mscoco.org/dataset/#detections-challenge2016>_ Dataset.
    ----------------------------------- Notes ----------------------------------
    Dataset class read Coco 201x like annotation json

    return image (PIL image object) , target (dict)

    The Target object is dict of annotation for one imagery
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
        Target [(Batch_Size),W,H]
    Use Mode Key to Init DataSet return content
    """

    def __init__(self, root, annFile,train=True, transforms=None,Mode='InstanceSegmentation',debug=False):
        super(CocoDataset, self).__init__()
             
        self.debug=debug
        self.root=root
        if train:
            self.image_root=os.path.join(self.root,"train2014/")
        else:
            self.image_root=os.path.join(self.root,"val2014/")
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.mode=Mode
        self.transforms=transforms
   
        print("# ---------------------------------------------------------------------------- #")
        print("#                            COCO DataSet Init Done                            #")
        print("# ---------------------------------------------------------------------------- #")
        print("# ====Root ",self.root)

    def checktarget(self,index):

        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        Ann = self.coco.loadAnns(ann_ids)
        labels=np.array([t["category_id"] for t in Ann])
        if len(labels)==0:
            return self.checktarget(index+1)
        else:
            return img_id,Ann,labels

        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id,Ann,labels= self.checktarget(index)
        
            
        
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.image_root, path)).convert('RGB')
        (w,h)=img.size
        boxes=np.array([t["bbox"] for t in Ann])
            
        
        instancemasks=np.array([self.annToMask(t,h,w) for t in Ann])
        segmask=np.zeros((h,w),dtype=np.int64)
        for index,t in enumerate(instancemasks):
            max_=max(segmask.max(),(t*labels[index]).max())
            segmask=segmask+t*labels[index]
            segmask[segmask>max_]=max_
        if self.debug:
            plt.imshow(segmask),plt.show()
        
        if self.mode=="InstenceSegmentation":
            target={
                "boxes":boxes,
                "labels":labels,
                "masks":instancemasks
            }
            
        if self.mode=="Detection":
            target={
                "boxes":boxes,
                "labels":labels
            }
        if self.mode=="Segmentation":
            target={
                "labels":labels,
                "masks":segmask
            }
            
            
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

    # ------------------------------ DataSetFunction ----------------------------- #

        def annToRLE(self, ann, height, width):
            """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle



# ---------------------------------------------------------------------------- #
#                                Coco evaluation                               #
# ---------------------------------------------------------------------------- #

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "coco"),
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def CocoEvaluation(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding COCO image IDs.
    coco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image])[0]
        t_prediction += (time.time() - t)

        # Convert results to COCO format
        image_results = build_coco_results(dataset, coco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           r["scores"], r["masks"])
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = coco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)



def main():
    AnnaFile="Data/Toolkit/dataset/annotation.json"
    DataSetRoot="Data/Toolkit/dataset/train2014"
    DataSets=CocoDataset(DataSetRoot,AnnaFile)
    for da in DataSets:
        print(da)



if __name__ == '__main__':
    main()
    