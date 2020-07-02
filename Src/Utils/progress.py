import numpy as np


import sys
import re
import rich
from rich.console import Console
from rich.table import Column, Table
import random
import numpy as np
from time import sleep

class ProgressRich():
    def __init__(self,max):
        self.console = Console()
        self.max=max
        
    def __call__(self,step,datadict=None):
        """
        update step of progress
        """
        self.datadict=datadict
        self.currentsetp=step
        self.formatdict()
        self.console.print(self.table,end='\n')
        
    
    
    
    
    def formatdict(self):
        """
        process table
        """
        self.table = Table(show_header=True, header_style="bold magenta")
        self.table.add_column("Date", style="dim", width=12)
        self.table.add_column("Title")
        for key in self.datadict.keys():
            value=self.datadict[key]
            self.table.add_row(
                key,
                str(value)
            )







def main():
    p=ProgressRich(100)
    from rich.progress import Progress

    with Progress() as progress:
        task3 = progress.add_task("[cyan]Cooking...", total=100)
        for i in range(100):
            progress.update(task3, advance=0.5)
            loss={
            "loss":np.random.randint(1, 100),
            "acc":np.random.randint(1, 100),
            "recall":np.random.randint(1, 100)
            }
            p(i,datadict=loss)
            sleep(0.3)


if __name__ == '__main__':
    main()
    