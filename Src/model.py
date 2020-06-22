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

print("# ----------------------------- Register the SRC ----------------------------- #")

import os
import sys
srcroot=sys.path[0][:-3]
sys.path.append(srcroot)
for i in sys.path:
    print("# ===== root ,",i)
print("# ----------------------------- Register the SRC ----------------------------- #")




from Src.dataset import dataset



class model(dataset):
    def __init__(self):
        dataset.__init__(self)
        print("# ---------------------------------------------------------------------------- #")
        print("#                               model init start                               #")
        print("# ---------------------------------------------------------------------------- #")




        print("# ---------------------------------------------------------------------------- #")
        print("#                               model init done                                #")
        print("# ---------------------------------------------------------------------------- #")
def main():
    TrainInstance=model()





if __name__ == '__main__':
    main()
    