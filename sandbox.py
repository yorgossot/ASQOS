import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import skeleton as ys

#Atoms. One in each cavity 
#Can support up to 3 atoms + 1 aux 
#Issue: use of not sparse matrices in gs_hamiltonian and e1_hamiltonian

# x is aux 
# o is Borregaard atom
# - is fiber

system_string = 'o-x-o-o-o'

sys = ys.system(system_string)


print(f'System total dimension:  {sys.dim}')

sys.construct_hamiltonian()

#print(sys.hamiltonian)
#print(sys.states)
#print(sys.excitations)
sys.construct_gs_hamiltonian()
print('\n \n --------Ground state------\n')
#print(sys.gs_hamiltonian)
print(sys.gs_states)


print('\n \n --------1st excited state------\n')
sys.construct_e1_hamiltonian()
#print(sys.e1_hamiltonian)
print(sys.e1_states)
print(f'Excited subspace dimension: {len(sys.e1_states)}')