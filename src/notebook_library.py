#
# File containing some extra functions to be used by the notebooks.
#
import  itertools,math,os, sys, copy, time,json, gc
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import sympy as sp
import qutip as qt
import seaborn as sns
import numpy as np
from importlib import reload  
import multiprocessing.pool as mp
from itertools import product
from IPython.display import clear_output, display
from pprint import pprint
import matplotlib.style
matplotlib.style.use('default')


def set_plot_big(plot_big):
    '''
    Makes plotting bigger
    '''
    if plot_big:
        params = {'legend.fontsize': 'xx-large',
            'figure.figsize': (12, 8),
            'axes.labelsize': 'xx-large',
            'axes.titlesize':'xx-large',
            'xtick.labelsize':'xx-large',
            'ytick.labelsize':'xx-large'}
        pylab.rcParams.update(params)
    else:
        matplotlib.style.use('default')


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



# istarmap.py for Python 3.8+
def custom_istarmap(self, func, iterable, chunksize=1):
    """
    starmap-version of imap
    https://stackoverflow.com/a/57364423
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mp.Pool._get_tasks(func, iterable, chunksize)
    result = mp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


