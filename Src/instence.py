# -*- coding: utf-8 -*-
# @Author: Winshare
# @Date:   2019-12-02 17:08:40
# @Last Modified by:   Winshare
# @Last Modified time: 2019-12-13 17:25:06

# Copyright 2019 Winshare
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

# ---------------------------------------------------------------------------- #
#                           Local STructure Reference                          #
# ---------------------------------------------------------------------------- #

from config_generator import cfg
from network_generator import NetworkGenerator
from dataset_generator import DatasetGenerator

# ---------------------------------------------------------------------------- #
#                            Local Module reference                            #
# ---------------------------------------------------------------------------- #
# 
# from general_train import train_one_epoch,evaluate
from Utils.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
import Utils.transforms as T
from Utils.coco_utils import ConvertCocoPolysToMask,_coco_remove_images_without_annotations
from Utils.coco_utils import get_coco_api_from_dataset
from Utils.coco_eval import CocoEvaluator
import Utils.utils as utils

# ---------------------------------------------------------------------------- #
#                           Official Module Reference                          #
# ---------------------------------------------------------------------------- #

from torch.utils.data import DataLoader
import torchvision
import torch
import os
import sys
import time
import math
import argparse



from torch.utils.tensorboard import SummaryWriter


root=os.path.abspath(__file__)
print('instence work on ',root)

class Instence(NetworkGenerator,DatasetGenerator):
    def __init__(self,
    instence_id=0,
    config_dir='./cfg',
    ):  

        # ---------------------------------------------------------------------------- #
        #                                workspace info                                #
        # ---------------------------------------------------------------------------- #

        self.root=root

        self.configfile=configfile
        print('root in :\n',os.path.join(self.root,'..'))
        sys.path.append(os.path.join(sys.path[0],'../'))
        print('workspace in:\n')
        for i in sys.path:
            print(i)
            
        DatasetGenerator.__init__(self)
        super(Instence,self).__init__()
        print('\n\n-----Instence Class Init-----\n\n')

        # ---------------------------------------------------------------------------- #
        #                                  dataloader                                  #
        # ---------------------------------------------------------------------------- #

        # ------------------------------ dataset object ------------------------------ #



        

        transforms=[]
        transforms.append(ConvertCocoPolysToMask())
        transforms.append(T.ToTensor())
        transforms.append(T.RandomHorizontalFlip(0.5))
        self.transform_compose=T.Compose(transforms)

        # ---------------------------------------------------------------------------- #
        #                                   temp part                                  #
        # ---------------------------------------------------------------------------- #




        if self.DefaultDataset:
            self.datasets=DatasetGenerator(transforms=self.transform_compose)
            self.datasets.DefaultDatasetFunction()

            self.trainset=_coco_remove_images_without_annotations(self.datasets.trainset)
            self.valset=self.datasets.valset
            print('-----train&val set already done')
 
        # ----------------------------- DataLoader object ---------------------------- #
        
        if self.DistributedDataParallel:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.trainset)
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.valset)
            print("-----DistributedDataParallel Sampler build done")
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=self.gpu_id)
            self.model_without_ddp = self.model.module
            

     



            

        if not self.DistributedDataParallel:

            self.train_sampler = torch.utils.data.RandomSampler(self.trainset)
            self.test_sampler = torch.utils.data.SequentialSampler(self.valset)
            print("-----DataSampler build done")





















            

        # ---------------------------------- Sampler --------------------------------- #
        
        if self.aspect_ratio_factor >= 0:
            self.group_ids = create_aspect_ratio_groups(self.trainset, k=self.aspect_ratio_factor)
            self.train_batch_sampler = GroupedBatchSampler(self.train_sampler,
            self.group_ids,
            self.BatchSize
            )
        else:
            self.train_batch_sampler = torch.utils.data.BatchSampler(
            self.train_sampler,
            self.BatchSize,
            drop_last=True
            )
        
        # ---------------------------------- loader ---------------------------------- #
        
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, 
            batch_sampler=self.train_batch_sampler,
            num_workers=self.worker_num,
            collate_fn=self.collate_fn)

        self.valloader = torch.utils.data.DataLoader(
            self.valset, 
            batch_size=self.BatchSize,
            sampler=self.test_sampler,
            num_workers=self.worker_num,
            collate_fn=self.collate_fn)







        



















        # ---------------------------------------------------------------------------- #
        #                               Instance Function                              #
        # ---------------------------------------------------------------------------- #

    def InstenceInfo(self):
        print('\n\n-----Start with Instence ID',self.InstanceID,'-----\n\n')
        self.Enviroment_Info()
        self.DatasetInfo()
        self.NetWorkInfo()


    def default_train(self):
        print('\n\n----- Start Training -----\n\n')
        start_time = time.time()

        # ---------------------------------------------------------------------------- #
        #                                  tensorboard                                 #
        # ---------------------------------------------------------------------------- #

        self.writer = SummaryWriter(log_dir=self.logdir,comment='experiment'+str(self.InstanceID))
        self.start=False

        
        
        if self.resume:
            assert os.path.exists(self.checkpoint),"Invalid resume model path"
            self.checkpoint=torch.load(self.checkpoint)
            self.model_without_ddp.load_state_dict(self.checkpoint['model'])
            self.optimizer.load_state_dict(self.checkpoint['optimizer'])
            self.lr_scheduler.load_state_dict(self.checkpoint['lr_scheduler'])
        baseloss=0
        for epoch in range(0,self.epochs):
            # ---------------------------------------------------------------------------- #
            #                                 epoch process                                #
            # ---------------------------------------------------------------------------- #
            sumloss=self.train_one_epoch(epoch)
            self.lr_scheduler.step()
            self.evaluate()
            if epoch==0:
                baseloss=sumloss
            if sumloss<baseloss:
                print("\n\n\n-----Model Update & Save")
                state = {"model":self.model.state_dict(), "optimizer":self.optimizer.state_dict(), 'epoch':epoch}
                torch.save(state, os.path.join(self.checkpoint,str(sumloss)+'.pth'))
            # ---------------------------------------------------------------------------- #
            #                                 epoch process                                #
            # ---------------------------------------------------------------------------- #






    def default_val(self):
        print('\n\n----- Val Processing -----\n\n')

    
    def inference(self):
        print('\n\n----- Inference Processing -----\n\n')

    def Evaluation(self):
        print('\n\n----- Evaluation Processing -----\n\n')
    
    
    def train_one_epoch(self ,epoch, print_freq=10):
        
        self.model.cuda()
        self.model.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)

        # lr_scheduler = None
        if epoch == 0:
            warmup_factor = 1. / 1000
            warmup_iters = min(1000, len(self.trainloader) - 1)

            # lr_scheduler = utils.warmup_lr_scheduler(self.optimizer, warmup_iters, warmup_factor)
    
        for images, targets in metric_logger.log_every(self.trainloader, print_freq, header):
            images,targets=self.todevice(images,targets)            
            loss_dict = self.model(images, targets)
            """
            {
                'loss_classifier': tensor(0.0925, device='cuda:0', grad_fn=<NllLossBackward>), 
                'loss_box_reg': tensor(0.0355, device='cuda:0', grad_fn=<DivBackward0>), 
                'loss_objectness': tensor(0.0270, device='cuda:0', grad_fn=<BinaryCrossEntropyWithLogitsBackward>), 
                'loss_rpn_box_reg': tensor(0.0112, device='cuda:0', grad_fn=<DivBackward0>)
            }
            """

            losses = sum(loss for loss in loss_dict.values())

            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            # if lr_scheduler is not None:
            self.lr_scheduler.step()

            metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
            metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])
            return losses


    def todevice(self,images,targets):
        """
        transform the local data to device 
        """
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return images,targets
        
    # ---------------------------------------------------------------------------- #
    #                                Writer function                               #
    # ---------------------------------------------------------------------------- #


    
        

    def _get_iou_types(self):
        model_without_ddp =self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            model_without_ddp = self.model.module
        iou_types = ["bbox"]

        # ------------------------------- for detection ------------------------------ #

        if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
            iou_types.append("segm")

        # ----------------------------- for segmentation ----------------------------- #

        if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
            iou_types.append("keypoints")

        # ------------------------------- for keypoint ------------------------------- #

        return iou_types


    @torch.no_grad()
    def evaluate(self,rate=0.1):
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        self.model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'

        coco = get_coco_api_from_dataset(self.valloader.dataset)
        iou_types = _get_iou_types(self.model)
        coco_evaluator = CocoEvaluator(coco, iou_types)

        for image, targets in metric_logger.log_every(self.valloader[:int(len(self.valloader)*rate)], 100, header):
            image = list(img.to(self.device) for img in image)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]


            torch.cuda.synchronize()
            model_time = time.time()
            outputs = self.model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        coco_evaluator.synchronize_between_processes()

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        torch.set_num_threads(n_threads)
        return coco_evaluator



def parser():
    parsers=argparse.ArgumentParser()
    parsers.add_argument("--config",default="Src/Config/Detection_Config_Template.json", help="dir of config file")
    args = parsers.parse_args()
    return args


def main():
    args=parser()
    global configfile
    configfile=args.config
    print(configfile)
    instence=Instence()
    # instence.default_train()



if __name__ == '__main__':
    main()
    