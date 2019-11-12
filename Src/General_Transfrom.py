import torchvision.transforms as T

class General_Transform(T):
    """
    The General_Transform class work for MultiMission Data Transform
    The Pytorch-like Transform just work for image data & different Mission
    have corresponding transform.
    
    
    Detection:

    The models expect a list of Tensor[C, H, W], in the range 0-1. 
    The models internally resize the images so that they have a minimum size of 800. 
    This option can be changed by passing the option min_size to the constructor of the models.

        * boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

        * labels (Int64Tensor[N]): the class label for each ground-truth box



    Segmentation:
    
    As with image classification models, all pre-trained models expect input images normalized in the same way. 
    The images have to be loaded in to a range of [0, 1] and then normalized using 
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]. 
    They have been trained on images resized such that their minimum size is 520.



    Instance_Segmantation:
    
    During inference, the model requires only the input tensors, and returns the post-processed predictions as a List[Dict[Tensor]], 
    one for each input image. The fields of the Dict are as follows:

        * boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W

        * labels (Int64Tensor[N]): the predicted labels for each image

        * scores (Tensor[N]): the scores or each prediction

        * masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range.
    
        In order to obtain the final segmentation masks, the soft masks can be thresholded, generally with a value of 0.5 (mask >= 0.5)



    BackBone:

    The models expect a list of Tensor[C, H, W], in the range 0-1. 



    Caption:




    KeyPoint:

    During training, the model expects both the input tensors, as well as a targets (list of dictionary), containing:
        * boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W
        * labels (Int64Tensor[N]): the class label for each ground-truth box
        * keypoints (FloatTensor[N, K, 3]): the K keypoints location for each of the N instances, in the format [x, y, visibility], where visibility=0 means that the keypoint is not visible.
    """
    def __init__(self,TransStrList=['ToTensor']):
        print('-----init General Transform Module-----')
        super(General_Transform,self).__init__()
        self.Mode=None


    def __DetCall__(self,image,target):
        """
        * boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W
        * labels (Int64Tensor[N]): the class label for each ground-truth box
        """
        pass



    def __Segcall__(self,image,target):
        """
        The images have to be loaded in to a range of [0, 1] and then normalized using 
        mean = [0.485, 0.456, 0.406] 
        std = [0.229, 0.224, 0.225]. 
    They have been trained on images resized such that their minimum size is 520.
        """
        pass

    
    
    def __MaskCall__(self,image,target):
        """
        * boxes (FloatTensor[N, 4]): the predicted boxes in [x1, y1, x2, y2] format, with values between 0 and H and 0 and W
        * labels (Int64Tensor[N]): the predicted labels for each image
        * scores (Tensor[N]): the scores or each prediction
        * masks (UInt8Tensor[N, 1, H, W]): the predicted masks for each instance, in 0-1 range.
        """
        pass

    def __CaptionCall__(self,image,target):
        pass




    def __call__(self,image,target):
        




def main():







if __name__ == '__main__':
    main()
    