import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import sage.all as sg

import system as ys


#Atoms. One in each cavity 
#Can support up to 5 atoms + 1 aux (takes some time)

#Current computational bottleneck: State and state excitation construction

# x is aux in a cavity
# o is Borregaard atom in a cavity
# - is fiber, in the end connects with the first cavity



system_string = 'x-o-o-o'

sys = ys.system(system_string)

print(f'System total dimension:  {sys.dim}')

sys.construct_gs_hamiltonian()
print('\n \n --------Ground state------\n')
print(sys.gs_hamiltonian)
print(sys.gs_states)
print(f'Ground subspace dimension: {len(sys.gs_states)}')

print('\n \n --------1st excited state------\n')
sys.construct_e1_hamiltonian()
print(sys.e1_hamiltonian)
#print(sys.e1_states)
print(f'Excited subspace dimension: {len(sys.e1_states)}')
