print("# ----------------------------- Register the SRC ----------------------------- #")

import os
import sys
srcroot=sys.path[0][:-3]
sys.path.append(srcroot)
for i in sys.path:
    print("# ===== root ,",i)
print("# ----------------------------- Register the SRC ----------------------------- #")

# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    model.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <winshare@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/06/23 14:40:30 by winshare          #+#    #+#              #
#    Updated: 2020/06/23 14:40:30 by winshare         ###   ########.fr        #
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




# ---------------------------- official reference ---------------------------- #


import argparse


# ------------------------------ local reference ----------------------------- #

from Src.dataset import dataset









class model(dataset):
    def __init__(self,cfg,process):
        self.config=cfg
        self.process=process
        dataset.__init__(self)
        print("# ---------------------------------------------------------------------------- #")
        print("#                               model init start                               #")
        print("# ---------------------------------------------------------------------------- #")




        print("# ---------------------------------------------------------------------------- #")
        print("#                               model init done                                #")
        print("# ---------------------------------------------------------------------------- #")

    def train(self):
        """
        subprocess do open tensorboard or not
        """
        import subprocess
        subprocess.run(["watch","-d","-n","0.1","nvidia-smi"])




def parser():
    parsers=argparse.ArgumentParser()
    parsers.add_argument("-config",default="Config/InstanceSegmentation/maskrcnn.json", help="dir of config file")
    parsers.add_argument("-process",default="train", help="(train_val) / (eval) / (inference) ")
    
    args = parsers.parse_args()
    return args



def main():
    args=parser()
    Model=model(cfg=args.config,process=args.process)
    Model.train()





if __name__ == '__main__':
    main()
    