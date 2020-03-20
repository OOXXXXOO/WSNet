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

class CocoDataset():
    """`MS Coco Support <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
        Rebuild DataSet Class For: 
            * Detection
            * InstanceSegmentation
            * Segmentation
        Use Mode Key to Init
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None,Mode='InstanceSegmentation'):
        # super(CocoDataset, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.root=root
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.mode=Mode

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.mode=="InstanceSegmentation":
            pass
            # sem_seg=
            # boxes=
            # labels=[T["label"]  for T in target]

        if self.mode=="Detection":
            pass

        if self.mode=="Segmentation":
            pass

        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)



def main():
    AnnaFile="Data/Toolkit/dataset/annotation.json"
    DataSetRoot="Data/Toolkit/dataset/train2014"
    DataSets=CocoDataset(DataSetRoot,AnnaFile)
    print(DataSets[0])



if __name__ == '__main__':
    main()
    