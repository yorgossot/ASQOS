'''
Functions supporting system module.
'''

import qutip as qt
import numpy as np
from scipy.sparse import csr_matrix
import sympy as sp
import pickle
import multiprocessing as mp
import itertools


def enumerated_together(idx : tuple , bare_matrix : sp.Matrix) \
    ->tuple[tuple[int,int],  sp.Add | sp.Mul ] : 
    '''
    Enumerated version of the sympy function together(). 
    Used for multiprocessing purposes.
    '''
    return idx, sp.together(bare_matrix[idx])

def together_for_sympy_matrices( non_togethered_matrix : sp.Matrix , processes : int | None = None  ) -> sp.Matrix:
    '''
    Multiprocessed version of the sympy function together(), targeted to deal with matrix elements in parallel.
    '''
    n_rows , n_cols = non_togethered_matrix.shape
    togethered_matrix = sp.Matrix.zeros(n_rows , n_cols)
    
    with mp.Pool(processes) as pool:
        iterable = zip(  itertools.product( range(n_rows) , range(n_cols) ), itertools.repeat(non_togethered_matrix))
        for idx, togethered_element in pool.starmap( enumerated_together , iterable):
            togethered_matrix[idx] = togethered_element

    return togethered_matrix


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

def make_into_hermitian(A):
    '''
    Takes a symbolic array A and returns it as a Hermitian. Does not double diagonal elements.
    '''
    assert A.shape[0] == A.shape[1]
    dim = A.shape[0]
    ones_w_0diag = np.ones((dim,dim))
    np.fill_diagonal(ones_w_0diag , 0)

    return  A   + sp.matrix_multiply_elementwise(A , sp.Matrix(ones_w_0diag) ).H
