from unicodedata import name
from resources.system import system



import multiprocessing as mp
import sympy as sp
import functools
import itertools

def enumerated_together(idx : tuple , bare_matrix : sp.Matrix): 
    print(idx)
    return idx, sp.together(bare_matrix[idx])

def together_for_sympy_matrices( non_togethered_matrix : sp.Matrix , processes : int | None = None  ) -> sp.Matrix:
    
    n_rows , n_cols = non_togethered_matrix.shape
    togethered_matrix = sp.Matrix.zeros(n_rows , n_cols)

    with mp.Pool(processes) as pool:
        iterable = zip(  itertools.product( range(n_rows) , range(n_cols) ), itertools.repeat(non_togethered_matrix))
        for idx, togethered_element in pool.starmap( enumerated_together , iterable):
            togethered_matrix[idx] = togethered_element

    return togethered_matrix

if __name__ == '__main__':
    s1 = system.system('O-x-O')
    a = together_for_sympy_matrices(s1.nj_hamiltonian_inv )
    print('Finished')