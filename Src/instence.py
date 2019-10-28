from config_generator import*
from network_generator import*
from dataset_generator import*
import sys
root=os.path.abspath(__file__)
print('instence work on ',root)

class Instence(NetworkGenerator,DatasetGenerator):
    """
                 
                             
                 |-->Dataset-->|——training-array generator<----->|  
                 |             |——training-to DataLoader         |                  |——template-config-generator——>——>|
                 |                                               |<-->|——Config-----|——readconfig<————————————————————|  
 Instance[MODE]——|                                               |                  |     ^       
                 |                                               |                  |——configure instance             |       
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
        print('\n\n-----instence init-----\n\n')
        
    def InstenceInfo(self):
        print('\n\n\t\tStart with Instence ID',self.instence_id,'\n\n')
        self.ConfigInfo()
        self.DatasetInfo()
        # self.NetWorkInfo()


def main():
    print('\n',Instence.mro(),'\n')
    instence=Instence()
    instence.DefaultDetection()
    Model=instence.model
    instence.DefaultDataset()
    print(Model)
    print(instence[20])
    # Net=instence.model



if __name__ == '__main__':
    main()
    