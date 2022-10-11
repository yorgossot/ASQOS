'''
Functions supporting system module.
'''

import qutip 
import numpy as np
from scipy.sparse import csr_matrix
import sympy 
import multiprocessing as mp
import itertools


def enumerated_together(idx : tuple , bare_matrix : sympy.Matrix) \
    ->tuple[tuple[int,int],  sympy.Add | sympy.Mul ] : 
    '''
    Enumerated version of the sympy function together(). 
    Used for multiprocessing purposes.
    '''
    return idx, sympy.together(bare_matrix[idx])

def together_for_sympy_matrices( non_togethered_matrix : sympy.Matrix , processes : int | None = None  ) -> sympy.Matrix:
    '''
    Multiprocessed version of the sympy function together(), targeted to deal with matrix elements in parallel.
    '''
    n_rows , n_cols = non_togethered_matrix.shape
    togethered_matrix = sympy.Matrix.zeros(n_rows , n_cols)
    
    with mp.Pool(processes) as pool:
        iterable = zip(  itertools.product( range(n_rows) , range(n_cols) ), itertools.repeat(non_togethered_matrix))
        for idx, togethered_element in pool.starmap( enumerated_together , iterable):
            togethered_matrix[idx] = togethered_element

    return togethered_matrix


def zero_operator(dim_list):
    '''
    Returns an operator with zero entries for a given dimension list.
    '''
    tensor_list = []
    for i in dim_list:
        tensor_list = qutip.identity(dim_list) 
    return 0*qutip.tensor(tensor_list)

def embed_ketbras_in_system( ketbras : list[qutip.Qobj], indices_in_system : list[int], dimensions : list[int] )-> qutip.Qobj:
    '''
    Embeds ketbra(s) in system of dimensions
    '''
    tensor_list = eye_operator_list(dimensions)
    
    if not isinstance(ketbras,list):
        index_in_system = indices_in_system
        ketbra = ketbras
        tensor_list[index_in_system] =  ketbras
    else:
        for ketbra, index_in_system  in zip(ketbras,indices_in_system):
            tensor_list[index_in_system] =  ketbra

    interaction_Qobj = qutip.tensor(tensor_list)
    return interaction_Qobj



def eye_operator_list(dim_list):
    '''
    Returns an identity operator list for a given dimension list. Used for tensor products.
    '''
    tensor_list = []
    for  dim in dim_list:
        tensor_list.append(qutip.identity(dim))   
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

    return  A   + sympy.matrix_multiply_elementwise(A , sympy.Matrix(ones_w_0diag) ).H
