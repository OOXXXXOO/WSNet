# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    dataset.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <winshare@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/06/22 16:43:43 by tanwenxuan        #+#    #+#              #
#    Updated: 2020/07/09 11:48:10 by winshare         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

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


from Src.network import network
import rich






class dataset(network):
    def __init__(self):
        network.__init__(self)
        print("# ---------------------------------------------------------------------------- #")
        print("#                              dataset init start                              #")
        print("# ---------------------------------------------------------------------------- #")


        self.mission_datasets=self.datasets_function_dict[self.MissionType]
        self.datasets_function=self.mission_datasets[self.DataSetType]
        self.datasets=self.datasets_function(self.MissionType,self.DataSet_Root)
        





















        print("# ---------------------------------------------------------------------------- #")
        print("#                              dataset init done                               #")
        print("# ---------------------------------------------------------------------------- #")