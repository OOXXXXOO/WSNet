# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    data_bridge.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/03/10 15:29:29 by winshare          #+#    #+#              #
#    Updated: 2020/03/11 14:17:00 by winshare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

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


# ---------------------------------------------------------------------------- #
#        This Script work for transform the data from one port to other        #
# ---------------------------------------------------------------------------- #

import torch
import numpy as np



class bridge():
    def __init__(self):
        print('----- data bridge activate ')
        """
        The data bridge work for transform the different data exchange port in specific format
        like :
        model output->loss function format

        BackBone:

            Input always is (N,C,W,H) Tensor ,BackBone (CNN) for Classification


            BackBone -> Segmentation
            
            BackBone -> Detection

            BackBone -> InstanceSegmentation

            BackBone -> Caption

            BackBone -> KeyPoint
            
        Segmentation:

            Segmentation DataDict
            {
                Official Segmentation Network
                Custom Segmentation Network
            }
            Segmentation Mask Label
            {
                Official Segmentation Network
                Custom Segmentation Network
            }

            inference -> ndarray
            
            inference -> Image

        Detection :

            Detection Datadict ->{
                Official RCNN Series
                Custom Detection Network
            }

            inference -> ndarray （np data vector）
            
            inference -> Visualization(Image)

        Instance :
            Instance Datadict ->{
                Official MaskRCNN Series
            }

        """
        def __init__(self):
            print("----------------------------- data bridge init -----------------------------")

    
