# Project Notes



Update Log:


MaskRCNN 要求

During training, the model expects both the input tensors, as well as a targets (list of dictionary),
containing:

    - boxes (FloatTensor[N, 4]): the ground-truth boxes in [x1, y1, x2, y2] format, with values
        between 0 and H and 0 and W
    
    - labels (Int64Tensor[N]): the class label for each ground-truth box
    
    - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance
        
1.把Segmentation字段转换为Mask


2.构建
target={
    boxes:[]
    labels:[]
    masks:[]
}


2020-3-20

重新设计

取消自定义Transform功能

设计新的固定Transform流程
