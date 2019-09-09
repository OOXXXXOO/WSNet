import os
import sys

################################################################################################
print('append sys path in ',os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
################################################################################################
#Process the enviroment path


import pandas
from torch.utils.data import Dataset, DataLoader



from Src.config_generator import*

class WSData(Dataset,cfg):
    def __init__(self):
        super(WSData,self).__init__()
        print('-----WSData Class Init-----')
    
    def Cust






def main():
    ws=WSData()
    ws.ConfigInfo()
    # WS.GenerateDefaultConfig()



if __name__ == '__main__':
    main()
    