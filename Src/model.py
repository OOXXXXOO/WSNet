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
import time
from tqdm import tqdm
import random
# ------------------------------ local reference ----------------------------- #

from Src.dataset import dataset




def animation(step,width=81):
    position=step%(width-1)
    line1=list("#"*width+"\n")
    line2=list("#"*width+"\n")
    line3=list("#"*width+"\n")
    line4=list("#"*width+"\n")
    line5=list("#"*width+"\n")
    if step>2:
        line1[position-2]=">"
        line2[position-1]=">"
        line3[position]=">"
        line4[position-1]=">"
        line5[position-2]=">"
    line1="".join(line1)
    line2="".join(line2)
    line3="".join(line3)
    line4="".join(line4)
    line5="".join(line5)
    print(line1,line2,line3,line4,line5)



    





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
        # subprocess.run(["watch","-d","-n","0.1","nvidia-smi"])
        for i in tqdm(range(1000)):
            time.sleep(1)




def parser():
    parsers=argparse.ArgumentParser()
    parsers.add_argument("-config",default="Config/InstanceSegmentation/maskrcnn.json", help="dir of config file")
    parsers.add_argument("-process",default="train", help="(train_val) / (eval) / (inference) ")
    
    args = parsers.parse_args()
    return args



def main():
    for i in range(1000):
        animation(i)
        time.sleep(0.1)
        os.system("clear")
    # args=parser()
    # Model=model(cfg=args.config,process=args.process)
    # Model.train()





if __name__ == '__main__':
    main()
    