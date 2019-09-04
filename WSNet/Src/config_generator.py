import pandas as pd
import os

class cfg():
    def __init__(self):
        print('\n\n-----Configure Generator Class Init -----\n\n')

    def ConfigInfo(self):
        print('\n-------------------------------------------GPU info:\n')
        os.system('nvidia-smi')
        print('\n-------------------------------------------GPU info:\n')

    