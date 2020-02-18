<!--
 Copyright 2020 winshare
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

# DataToolkit Document


### labelmejson transform to coco annotation

### usage:

` labelme2coco.py [-h] [--a A] [--l L] [--o O]`

#### optional arguments:

      -h, --help  show this help message and exit
      --a A       dir of anno file like ./annotation/
      --l L       dir of label name file like label.txt
      --o O       dir of output annotation file like annotation.json

#### example:

```bash
python labelme2coco.py --l labels.txt --a ./labelme/demo/train2014/ --o annotation.json
```

