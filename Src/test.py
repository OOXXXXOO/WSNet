# -*- coding: utf-8 -*-
# @Author: Winshare
# @Date:   2019-12-12 16:58:01
# @Last Modified by:   Winshare
# @Last Modified time: 2019-12-12 17:08:19
import json 


def main():
    print("Hello, World!")
    jsons='/media/winshare/98CA9EE0CA9EB9C8/COCO_Dataset/annotations/instances_train2014.json'
    with open(jsons,'r') as f:
        content=json.load(f)
        newcontent={"images":content['images'][0:100]}
    with open('COCOFormat.json','w') as fpp:   
        json.dump(newcontent,fpp)


if __name__ == "__main__":
    main()
