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
#    instance.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <winshare@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/18 15:22:43 by winshare          #+#    #+#              #
#    Updated: 2020/02/18 15:22:43 by winshare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #




import sys
import os
from dataset import DATASET

class INSTANCE(DATASET):
    def __init__(self):
        self.configfile="Config/Demo.json"
        DATASET.__init__(self)
        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #





        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #
        print("\n\n---------------------- INSTANCE Class Init Successful ----------------------\n\n")

A=INSTANCE()