from instence import Instence
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def trainval():
    print('-----Train Process')
    train=Instence()    
    train.InstenceInfo()
    train.DefaultDataset(Mode='train')
    val=Instence()
    val.DefaultDataset(Mode='val')

    ##### Network Init
    if train.usegpu:
        train.model.to(train.device)




    ##### Loader Init

    trainloader=DataLoader(train,
    train.BatchSize,
    shuffle=True,
    num_workers=train.worker_num,
    collate_fn=train.detection_collate_fn
    )

    valloder=DataLoader(val,val.BatchSize,shuffle=True,num_workers=val.worker_num)


    ##### train process 
    train.model.train()
    
    for epoch in range(train.epochs):
        print("\n\n---------Epoch:",epoch)
        for index,(images,targets) in enumerate(trainloader):
            if train.usegpu:
                images = list(image.to(train.device) for image in images)
                targets = [{k: v.to(train.device) for k, v in t.items()} for t in targets]
                if len(targets[0]['boxes'])==0:
                    continue
                train.Optimzer.zero_grad()
                loss_dict=train.model(images,targets)
                losses = sum(loss for loss in loss_dict.values())
                loss=losses.cpu().detach().numpy()
                print('-----Step',index,'--LOSS--',loss)
                losses.backward()
                train.Optimzer.step()
            if not train.usegpu:
                exit(0)
        evaluate(train.model,valloder,train.device)




##### eval & model IO process
@torch.no_grad()
def evaluate(model, data_loader, device):
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
    trainval()
if __name__ == '__main__':
    main()
    