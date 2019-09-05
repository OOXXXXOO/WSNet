from config_generator import*
from network_generator import*
from dataset_generator import*



class Instence(cfg,NetworkGenerator,DatasetGenerator):
    """
                                  |——template-config-generator——>——>|
                    |——Config-----|——readconfig<————————————————————|  
                    |     ^       |——configure instance             |  
                    |     |                                         |               
    Instance[MODE]——|-->Dataset-->|——training-array generator       |  
                    |             |——training-to DataLoader         |  
                    |                                               |          
                    |——Network----|——readconfig<————————————————————|  
                                  |——Network Generator
                                  |——Network Process——————>——————————>Train/Val/Test

    MODE=[Segmentation,Detection,Mask]
    """
    def __init__(self,
    instence_id=0,
    config_dir='./cfg',
    ):
        # super(NetworkGenerator,self).__init__()
        # super(DatasetGenerator,self).__init__() 
        # super(cfg,self).__init__()
        super(Instence,self).__init__()
        print('\n\n-----instence init-----\n\n')
        
    def InstenceInfo(self):
        print('\n\n\t\tStart with Instence ID',self.instence_id,'\n\n')
        self.ConfigInfo()
        self.DatasetInfo()
        self.NetWorkInfo()
        
print('\n',Instence.mro(),'\n')
instence=Instence()
instence.DefaultDetection()
# Net=instence.model
