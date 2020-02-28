# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    instance.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: winshare <tanwenxuan@live.com>             +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2020/02/28 11:46:08 by winshare          #+#    #+#              #
#    Updated: 2020/02/28 12:21:35 by winshare         ###   ########.fr        #
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






import sys
import os
from dataset import DATASET
class INSTANCE(DATASET):
    def __init__(self,cfg):
        self.configfile=cfg
        DATASET.__init__(self,cfg=cfg)
        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #









        # -------------------------------- DataLoader -------------------------------- #
        print("-------------------------------- DataLoader --------------------------------")






















        # ---------------------------------------------------------------------------- #
        #                                 init process                                 #
        # ---------------------------------------------------------------------------- #
        print("\n\n---------------------- INSTANCE Class Init Successful ----------------------\n\n")



def parser():
    parsers=argparse.ArgumentParser()
    parsers.add_argument("--config",default="Src/Config/Demo.json", help="dir of config file")
    args = parsers.parse_args()
    return args



def main():
    args=parser()
    configfile=args.config
    print(configfile)
    instence=Instence(configfile=configfile)
    instence.default_train()



if __name__ == '__main__':
    main()
    