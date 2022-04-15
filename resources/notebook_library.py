#
# File containing some extra functions to be used by the notebooks.
#
import tqdm, itertools,math,os, sys, copy, time,json, quantiphy
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import sage.all as sg
import seaborn as sns
import numpy as np
from importlib import reload  
import multiprocessing.pool as mp
from itertools import product
from IPython.display import clear_output, display
import matplotlib.style
matplotlib.style.use('default')
params = {'legend.fontsize': 'xx-large',
          'figure.figsize': (15, 10),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

with  open('resources/experimental_values.json') as json_file: 
    experimental_values_dict = json.load(json_file) 

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


# istarmap.py for Python 3.8+
def istarmap(self, func, iterable, chunksize=1):
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
mp.Pool.istarmap = istarmap

from sage.symbolic.expression_conversions import ExpressionTreeWalker # for the simplif funct

def MMA_simplify( expr , full=False):
    '''
    Simplifies expression with the use of Mathematica and returns the result in SageMath.
    Can be used for both matrices and symbolic expressions.
    '''
    if full==True:
        expr_simpl_m  =  expr._mathematica_().FullSimplify()
    else:
        expr_simpl_m  =  expr._mathematica_().Simplify()

    expr_simpl_sg = expr_simpl_m._sage_()
    
    if  type(expr_simpl_sg) is list:
        #the expression is a matrix. needs a modification
        expr_simpl_sg = sg.matrix(expr_simpl_sg)
    
    return expr_simpl_sg



def is_integer_num(n):
    '''
    Is used in the next class.
    https://note.nkmk.me/en/python-check-int-float/
    '''
    if isinstance(n, int):
        return True
    else:
        return n.is_integer()
    return False

class symround_class(ExpressionTreeWalker):
    def __init__(self, **kwds):
        """
        A class that walks the tree and replaces numbers by numerical
        approximations with the given keywords to `numerical_approx`.
        EXAMPLES::
            sage: var('F_A,X_H,X_K,Z_B')
            sage: expr = 0.0870000000000000*F_A + X_H + X_K + 0.706825181105366*Z_B - 0.753724599483948
            sage: symround_class(digits=3)(expr)
            0.0870*F_A + X_H + X_K + 0.707*Z_B - 0.754

        Taken from:
        https://ask.sagemath.org/question/46059/is-it-possible-to-round-numbers-in-symbolic-expression/#46061
        """
        self.kwds = kwds
        self.digits_dic = {'digits' : self.kwds["digits"] }
        self.tol =  self.kwds["tol"] 
        self.show_del = self.kwds["show_del"] 

    def pyobject(self, ex, obj):
        if hasattr(obj, 'numerical_approx'):
            if hasattr(obj, 'parent'):               
                if obj in sg.RealField(): 
                    obj = sg.real(obj).numerical_approx(**self.digits_dic)
                    #simplify real numbers 
                    #dont spoil integers
                    if obj.parent()==sg.IntegerRing():
                        return obj
                    #if a float is integer, transform it into sg.Integer
                    if obj.is_integer():
                        return sg.Integer(obj)
                    #if a float is too small, delete it
                    if abs(obj)<self.tol:
                        if self.show_del: print(f'symround: Deleted coefficient {obj}')
                        return 0                
                else:
                    #simplify complex numbers
                    re = obj.real().numerical_approx(**self.digits_dic)
                    im = obj.imag().numerical_approx(**self.digits_dic)
                    if abs(re)<self.tol and re!=0:
                        if self.show_del: print(f'symround: Deleted coefficient {re}')
                        re = 0
                    if abs(im)<self.tol and im!=0:
                        if self.show_del: print(f'symround: Deleted coefficient {im}')
                        im = 0
                    #check if any of the im real parts is imaginary
                    if is_integer_num(re) and is_integer_num(im):
                        return sg.Integer(re) + sg.I * sg.Integer(im)
                    if is_integer_num(re):
                        return sg.Integer(re) + sg.I * im.numerical_approx(**self.digits_dic)
                    if is_integer_num(im):
                        return re.numerical_approx(**self.digits_dic) + sg.I * sg.Integer(im)           
                
            return obj.numerical_approx(**self.digits_dic)
        else:
            return obj


from collections.abc import Iterable
def symround(expr,digits=1, show_del= True , tol=1e-12 ):
    '''
    Uses symround to apply it to matrices.
    '''
    if isinstance(expr,Iterable):
        #expression is a matrix
        matr = expr
        nc, nr = matr.ncols(), matr.nrows()
        A = sg.copy(matr.parent().zero())
        for r in range(nr):
            for c in range(nc):
                A[r,c] = symround_class(digits=digits , tol=tol, show_del=show_del  )(matr[r,c])
        return A
    else:
        return symround_class(digits=digits , tol=tol, show_del=show_del)(expr)