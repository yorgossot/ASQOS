import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import sage.all as sg
from sage.symbolic.expression_conversions import ExpressionTreeWalker # for the simplif funct

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
    Not to be used for matrices.
    '''
    if full==True:
        expr_simpl_m  =  expr._mathematica_().FullSimplify()
    else:
        expr_simpl_m  =  expr._mathematica_().Simplify()
    
    expr_simpl_sg = expr_simpl_m._sage_()

    return expr_simpl_sg

def MMA_simplify_matr( matr , full=False):
    '''
    Simplifies square matrix with the use of Mathematica and returns the result in SageMath.
    '''
    if full==True:
        matr_simpl_m  =  matr._mathematica_().FullSimplify()
    else:
        matr_simpl_m  =  matr._mathematica_().Simplify()
    
    list_simpl_sg = matr_simpl_m._sage_()  #sagemath creates a list out of the matrix.


    matr_simpl_sg = sg.matrix(list_simpl_sg)

    return matr_simpl_sg



class symround(ExpressionTreeWalker):
    def __init__(self, **kwds):
        """
        A class that walks the tree and replaces numbers by numerical
        approximations with the given keywords to `numerical_approx`.
        EXAMPLES::
            sage: var('F_A,X_H,X_K,Z_B')
            sage: expr = 0.0870000000000000*F_A + X_H + X_K + 0.706825181105366*Z_B - 0.753724599483948
            sage: SubstituteNumericalApprox(digits=3)(expr)
            0.0870*F_A + X_H + X_K + 0.707*Z_B - 0.754

        Taken from:
        https://ask.sagemath.org/question/46059/is-it-possible-to-round-numbers-in-symbolic-expression/#46061
        """
        self.kwds = kwds

    def pyobject(self, ex, obj):
        if hasattr(obj, 'numerical_approx'):
            if hasattr(obj, 'parent'):
                if obj.parent()==sg.IntegerRing():
                    return obj
            return obj.numerical_approx(**self.kwds)
        else:
            return obj