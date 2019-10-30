from config_generator import*
from network_generator import*
from dataset_generator import*
from torch.utils.data import Dataset
import sys
root=os.path.abspath(__file__)
print('instence work on ',root)

class Instence(NetworkGenerator,DatasetGenerator,Dataset):
    """
                 
                             
                 |-->Dataset-->|——training-array generator<----->|  
                 |             |——training-to DataLoader         |                  |——template-config-generator——>——>|
                 |                                               |<-->|——Config-----|——readconfig<————————————————————|  
 Instance[MODE]——|                                               |                  |     ^       
                 |                                               |                  |——configure instance—————————————|       
                 |                                               |          
                 |——Network----|——readconfig<———————————————————>|  
                               |——Network Generator
                               |——Network Process——————>|
                                                        |---->Train/Val/Test

    MODE=[Segmentation,Detection,Instence,Caption]
    """
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

    def targetmap(self):
        """
        
        """
        pass
        
    def InstenceInfo(self):
        print('\n\n-----Start with Instence ID',self.InstanceID,'-----\n\n')
        self.Enviroment_Info()
        self.DatasetInfo()
        self.NetWorkInfo()


def main():
    
    instence=Instence()
    instence.DefaultDetection()
    instence.DefaultDataset()
    print(instence.Optimzer)
    print(instence.Loss_Function)
    print(instence.model)
    print(instence[20])
    # Net=instence.model



if __name__ == '__main__':
    main()
    