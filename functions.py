import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import sage.all as sg
from sage.symbolic.expression_conversions import ExpressionTreeWalker # for the simplif funct
import math as math
import pickle


def save_object(obj, filename):
    '''
    Saves object.
    https://stackoverflow.com/a/4529901
    '''
    with open(f'saved_objects/{filename}.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
def load_object(filename):
    '''
    Load object.
    https://stackoverflow.com/a/4529901
    '''
    with open(f'saved_objects/{filename}.pkl', 'rb') as input:
        return pickle.load(input)

def zero_operator(dim_list):
    '''
    Returns an operator with zero entries for a given dimension list.
    '''
    tensor_list = []
    for i in dim_list:
        tensor_list = qt.identity(dim_list) 
    return 0*qt.tensor(tensor_list)

def id_operator(dim_list):
    '''
    Returns an identity operator for a given dimension list.
    '''
    tensor_list = []
    for i in dim_list:
        tensor_list = qt.identity(dim_list) 
    return qt.tensor(tensor_list)

def id_operator_list(dim_list):
    '''
    Returns an identity operator list for a given dimension list. Used for tensor products.
    '''
    tensor_list = []
    for  dim in dim_list:
        tensor_list.append(qt.identity(dim))   
    return tensor_list

def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix


    Taken from: https://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:,mask]
    else:
        return mat


def elementwise(operator, M, N):
    '''
    SageMath elementwise operartion
    https://ask.sagemath.org/question/10707/element-wise-operations/
    '''
    assert(M.parent() == N.parent())
    nc, nr = M.ncols(), M.nrows()
    A = sg.copy(M.parent().zero())
    for r in range(nr):
        for c in range(nc):
            A[r,c] = operator(M[r,c], N[r,c])
    return A


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