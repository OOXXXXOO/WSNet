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
#    dataset.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <winshare@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/18 15:23:49 by winshare          #+#    #+#              #
#    Updated: 2020/02/18 15:23:49 by winshare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #



import sys 
print("---dataset.py workspace in :\n",sys.path)
from Utils.Transform.transform import Compose
from network import NETWORK

class DATASET(NETWORK):
    def __init__(self):
        NETWORK.__init__(self)
        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #





        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #
        print("\n\n----------------------- DATASET Class Init Successful ----------------------\n\n")

