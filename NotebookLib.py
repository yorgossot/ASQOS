#
# File containing some extra functions to be used by the notebooks.
#

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import sage.all as sg
import seaborn as sns
import numpy as np
from importlib import reload  
import math,os, sys
import time
import multiprocessing as mp
from itertools import product
from IPython.display import clear_output, display
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (15, 10),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)


def progressBar(current, total, barLength = 20):
    clear_output(wait=True)
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent))

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def ArgMax(npArray):
    return np.unravel_index(np.argmax(npArray), npArray.shape)

def ArgMin(npArray):
    return np.unravel_index(np.argmin(npArray), npArray.shape)


def NumElemsInList(element):
    count = 0
    if isinstance(element, list):
        for each_element in element:
            count += NumElemsInList(each_element)
    else:
        count += 1
    return count

class Time:
    def __init__(self,NameOfProcedure):
        self.NameOfProcedure = NameOfProcedure
        
    def __enter__(self):
        self.tic = time.time()
        

    def __exit__(self, exc_type, exc_val, exc_tb):
        ElapsedTime = time.time() - self.tic
        print(self.NameOfProcedure + f' took {ElapsedTime} to complete.')
