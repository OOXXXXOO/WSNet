import sys
from pycocotools.coco import COCO
# from Data.COCO.utils import *
# from Data.COCO.references.detection.coco_utils import get_coco as detcoco
# from Data.COCO.references.segmentation.coco_utils import get_coco as segcoco

class CocoDataSet(COCO):
    def __init__(self,mission,root,transforms=[],sets='train',version=2014,mode="instance"):
        """
        * mission :
                Detection 
                Segmentation 
                InstanceSegmentation
                KeyPoint
        * root 
         |---Root
                |---annotations
                        |---instance_train2014.json
                        |---instance_val2014.json
                |---train2014
                        |---images...
                        ...
                |---val2014
                        |---images...
                        ...
                |---test2014
                        |---images...
                        ...
        * mode :instances,captions,person_keypoints
        * version :2014,2017
        * set : "train","val","test"
        """
        
        print(
            "------------------------------------------------------------\n"
            "          _             _             _             _       \n"
            "        /\ \           /\ \         /\ \           /\ \     \n"
            "       /  \ \         /  \ \       /  \ \         /  \ \    \n"
            "      / /\ \ \       / /\ \ \     / /\ \ \       / /\ \ \   \n"
            "     / / /\ \ \     / / /\ \ \   / / /\ \ \     / / /\ \ \  \n"
            "    / / /  \ \_\   / / /  \ \_\ / / /  \ \_\   / / /  \ \_\ \n"
            "   / / /    \/_/  / / /   / / // / /    \/_/  / / /   / / / \n"
            "  / / /          / / /   / / // / /          / / /   / / /  \n"
            " / / /________  / / /___/ / // / /________  / / /___/ / /   \n"
            "/ / /_________\/ / /____\/ // / /_________\/ / /____\/ /    \n"
            "\/____________/\/_________/ \/____________/\/_________/     \n"
            "------------------------------------------------------------\n"
        )

        print("# ===== COCO DataSet Build with mission :%s"%mission)
        anno_file_template = "{}_{}"+str(version)+".json"
        anno=os.path.join("annotations", anno_file_template.format(mode, sets))
        self.abs_anno=os.path.join(root,anno)
        print("# ===== annotation :",self.abs_anno)
        self.dataset=None
        # if mission=="segmentation":
        #     self.dataset=detcoco(root,sets,transforms)
        # else:
        #     self.dataset=segcoco(root,sets,transforms)
        """
        COCO Sucess & fix the transform
        """
            
    def __call__(self):
        return self.dataset


def main():
    COCORoot="/workspace/COCO/"
    C=CocoDataSet(
        root=COCORoot,
        mission="InstanceSegmentation",
        version=2014,
        mode="instance"
    )    




if __name__ == '__main__':
    main()
    