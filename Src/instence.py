from config_generator import cfg
from network_generator import NetworkGenerator
from dataset_generator import DatasetGenerator
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import sys
root=os.path.abspath(__file__)
print('instence work on ',root)

class Instence(NetworkGenerator,DatasetGenerator,Dataset):
    def __init__(self,
    instence_id=0,
    config_dir='./cfg',
    ):  
        self.root=root
        print('root in :\n',os.path.join(self.root,'..'))
        sys.path.append(os.path.join(sys.path[0],'../'))
        print('workspace in:\n')
        for i in sys.path:
            print(i)
        
        DatasetGenerator.__init__(self)
        # super(NetworkGenerator,self).__init__()
        super(Instence,self).__init__()
        print('\n\n-----Instence Class Init-----\n\n')

        #####################################################
        #Dataloader
        
        self.TrainSet=DatasetGenerator()
        self.TrainSet.DefaultDataset(Mode='train')
        self.Trainloader=DataLoader(
            self.TrainSet,
            self.BatchSize,
            shuffle=True,
            num_workers=self.worker_num,
            collate_fn=self.TrainSet.detection_collate_fn
        )
        self.ValSet=DatasetGenerator()
        self.ValSet.DefaultDataset(Mode='val')
        self.Valloader=DataLoader(
            self.ValSet,
            self.BatchSize,
            shuffle=True,
            num_workers=self.worker_num,
            collate_fn=self.ValSet.detection_collate_fn
        )
        #######################################################

    def ToDecive(self,images,targets):
        images = list(img.to(self.device) for img in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        return images,targets


    def targetmap(self):
        """
        
        """
        pass
        
    def InstenceInfo(self):
        print('\n\n-----Start with Instence ID',self.InstanceID,'-----\n\n')
        self.Enviroment_Info()
        self.DatasetInfo()
        self.NetWorkInfo()
    def train(self):
        print('\n\n----- Start Training -----\n\n')

        self.model.cuda()
        

        #############################################################
        #11-28
        """
        IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1
        """
        for epoch in range(self.epochs):
            print('---Epoch : ',epoch)
            for index,(images,targets) in enumerate(self.Trainloader):
                images,targets=self.ToDecive(images,targets)
                self.optimizer.zero_grad()
                loss_dict=self.model(images,targets)
                losses = sum(loss for loss in loss_dict.values())
                loss=losses.cpu().detach().numpy()
                print('-----Step',index,'--LOSS--',loss)
                losses.backward()
                self.optimizer.step()
        

    def val(self,valloader):
        print('\n\n----- Val Processing -----\n\n')

    
    def inference(self):
        print('\n\n----- Inference Processing -----\n\n')

    def Evaluation(self):
        print('\n\n----- Evaluation Processing -----\n\n')
    
    
    @torch.no_grad()
    def evaluate(self, data_loader):
        model=self.model
        device=self.device
        n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        model.eval()
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'

        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = _get_iou_types(model)
        coco_evaluator = CocoEvaluator(coco, iou_types)

        for image, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


            torch.cuda.synchronize()
            model_time = time.time()
            outputs = model(image)

            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time

            res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)





def main():
    
    instence=Instence()
    instence.train()



if __name__ == '__main__':
    main()
    