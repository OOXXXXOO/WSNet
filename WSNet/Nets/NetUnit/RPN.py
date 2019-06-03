import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F


"""
Features Extraction from the image.

Creating anchor targets.

Locations and objectness score prediction from the RPN network.

Taking the top N locations and their objectness scores aka proposal layer

Passing these top N locations through Fast R-CNN network and generating locations and cls predictions for each location is suggested in 4.

generating proposal targets for each location suggested in 4

Using 2 and 3 to calculate rpn_cls_loss and rpn_reg_loss.

using 5 and 6 to calculate roi_cls_loss and roi_reg_loss.
"""

class RPN(nn.Module):
    """RPN Network from CFG """
    def __init__(self):
        super(RPN,self).__init__()
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        """                                             
                          |-->rpn_cls_score_net--->_______--->Class Scores--->|softmax--->|Class Probabilities
                          |    w/16,h/16,9,2       reshape
                          |  
        rpn_net -->relu-->|
                          |  
                          |-->rpn_bbx_pred_net---->_______--->Bounding Box regressors---->|
                               w/16,h/16,9,4       reshape
        
        """
        # rpn_net -->relu
        self.RPN_Net=nn.Conv2d(self.din,512,3,1,1,bias=True)

        # rpn_cls_score_net # (3*3*2)-->(9*2)
        self.RPN_cls_score_out=len(self.anchor_scales)*len(anchor_ratios)*2
        self.RPN_cls_score_net=nn.Conv2d(512,self.RPN_cls_score_out,1,1,0)

        # rpn_bbx_pred_net #(3*3*4)-->(9*4)
        self.RPN_bbx_pred_out=len(self.anchor_scales)*len(anchor_ratios)*2
        self.RPN_bbx_pred_net=nn.Conv2d(512,self.RPN_cls_score_out,1,1,0)



    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x