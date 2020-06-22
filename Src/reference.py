# Copyright 2020 tanwenxuan
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
#    reference.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: tanwenxuan <tanwenxuan@student.42.fr>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/06/22 16:34:57 by tanwenxuan        #+#    #+#              #
#    Updated: 2020/06/22 16:34:57 by tanwenxuan       ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import torchvision.models as models
import torchvision.datasets as dataset
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


class reference():
    def __init__(self):

        print("# ---------------------------------------------------------------------------- #")
        print("#                          reference library init start                        #")
        print("# ---------------------------------------------------------------------------- #")

        # ---------------------------------------------------------------------------- #
        #                               DataSet Function                               #
        # ---------------------------------------------------------------------------- #

        # self.datasets_function_dict={
        #     "Classification":{
        #         "MINST":dataset.MNIST,
        #         "FashionMINST":dataset.FashionMNIST,
        #         "KMINST":dataset.KMNIST,
        #         "EMINST":dataset.EMNIST,
        #         "CIFAR10":dataset.CIFAR10,
        #         "CIFAR100":dataset.CIFAR100,
        #         "ImageNet":dataset.ImageNet
        #     },
        #     "Detection":{
        #         "CocoDetection":CocoDataset,
        #         "VOC_Detection":dataset.VOCDetection
        #     },
        #     "Segmentation":{
        #         "VOC_Segmentation":dataset.VOCSegmentation,
        #         "Cityscapes":dataset.Cityscapes,
        #         "Costum_NPY_DataSet":Costum_NPY_DataSet,
        #         "CocoSegmentation":CocoDataset
        #     },
        #     "Caption":{
        #         "CocoCaptions":dataset.CocoCaptions
        #     },
        #     "InstenceSegmentation":{
        #         "CocoDetection":CocoDataset
        #     }
        # }
        # self.dataset_support_list=self.datasets_function_dict.keys()



        # ---------------------------------------------------------------------------- #
        #                              Optimizer Function                              #
        # ---------------------------------------------------------------------------- #

        self.OptimDict={
           "SGD":optim.SGD,                                                                                                                                              
           "ASGD":optim.ASGD,
           "Adam":optim.Adam,
           "Adadelta":optim.Adadelta,
           "Adagrad":optim.Adagrad,
           "AdamW":optim.AdamW,
           "LBFGS":optim.LBFGS,
           "RMSprop":optim.RMSprop,
           "SparseAdam":optim.SparseAdam,
           "Adamax":optim.Adamax
        }





        print("# ---------------------------------------------------------------------------- #")
        print("#                              reference init done                             #")
        print("# ---------------------------------------------------------------------------- #")









def main():
    print("Unit Test for reference")







if __name__ == '__main__':
    main()
    