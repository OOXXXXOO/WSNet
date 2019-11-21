from config_generator import cfg
from network_generator import NetworkGenerator
from dataset_generator import DatasetGenerator
from torch.utils.data import Dataset
import sys
from torch.utils.data import DataLoader
import os
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
        #####
        #Epochs
        for epoch in range(self.epochs):
            print('---Epoch : ',epoch)

        

    def val(self,valloader):
        print('\n\n----- Val Processing -----\n\n')

    
    def inference(self):
        print('\n\n----- Inference Processing -----\n\n')

    def Evaluation(self):
        print('\n\n----- Evaluation Processing -----\n\n')




def main():
    
    instence=Instence()
    instence.train()



if __name__ == '__main__':
    main()
    