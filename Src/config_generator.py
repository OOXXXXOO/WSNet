import pandas as pd
import os
import json

class cfg():
    def __init__(self):
        print('\n\n-----Configure Generator Class Init -----\n\n')
        self.__defaultconfig='Src/config/config_template.json'
        self.__json=json.load(open(self.__defaultconfig))
        self.checkpoint=self.__json['']


    def GenerateDefaultConfig(self,mode='detection'):
        print('Generate Default Config with mode :',mode)
        # print(self.__json)




































    def print_dict(self,d,n=0):
        print('-----config dict-----')
        for k,v in d.items():
            print ('\t'*n)
            if type(v)==type({}):
                print("%s : {" % k)
                self.print_dict(v,n+1)
            else:
                print("%s : %s" % (k,v))
        if n!=0:
            print('\t'*(n-1)+ '}')

    def ConfigInfo(self):
        self.print_dict(self.__json)

        print('\n-------------------------------------------NVCC info:\n')
        os.system('nvcc -V')
        print('\n-------------------------------------------GPU info:\n')
        os.system('nvidia-smi')
        print('\n-------------------------------------------GPU info:\n')

    