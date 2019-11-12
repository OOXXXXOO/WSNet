from instence import *
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
    ##### Dataset init
    train=Instence()
    train.InstenceInfo()
    train.DefaultDataset(Mode='train')
    # val=Instence()
    # val.DefaultDataset(Mode='val')

    ##### Network Init
    if train.usegpu:
        train.model.to(train.device)

    ##### NetworkTest
    # train.model.eval()
    # in_=torch.randn((1,3,800,800),dtype=torch.float32)
    # in_=in_.to(train.device)
    # print("in:",in_,in_.size())
    # out_=train.model(in_)
    # print("out",out_)


    ##### Loader Init

    trainloader=DataLoader(train,train.BatchSize,shuffle=True,num_workers=train.worker_num)
    # valloder=DataLoader(val,val.BatchSize,shuffle=True,num_workers=val.worker_num)

    ##### criterion Init
    #
    ##### optimizer Init
    #
    ##### train process 
    
    for epoch in range(train.epochs):
        print("\n\n-----Epoch:",epoch)
        for inputs,targets in tqdm(trainloader):
            if train.usegpu:
                inputs,targets=inputs.to(train.device),targets.to(train.device)
                output=train.model(inputs)
                loss=train.Loss_Function(output,targets)
                train.Optimzer.zero_grad()
                loss.backward()
                train.Optimzer.step()
            if not train.usegpu:
                output=train.model(inputs)
                loss=train.Loss_Function(output,targets)
                train.Optimzer.zero_grad()
                loss.backward()
                train.Optimzer.step()
    
    ##### eval & model IO process
























def main():
    trainval()
if __name__ == '__main__':
    main()
    