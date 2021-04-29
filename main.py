import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import sage.all as sg
import time 
import system as ys


#Atoms. One in each cavity 
#Can support up to 5 atoms + 1 aux (takes some time)

#Current computational bottleneck: State and state excitation construction

# x is aux in a cavity
# o is Borregaard atom in a cavity
# - is fiber, in the end connects with the first cavity



system_string = 'x-o-o-o'
t0 = time.time()
sys = ys.system(system_string)
t1 = time.time()

print((f'\n\nTook   {round(t1-t0)}  to initialize.\n\n'))

print(f'System total dimension:  {sys.dim}')


print('\n \n --------Ground state subspace------\n')
#print(sys.gs_hamiltonian)
print(sys.gs_states)
print(f'Ground subspace dimension: {len(sys.gs_states)}')

print('\n \n --------1st excited state subspace------\n')

#print(sys.e1_hamiltonian)
print(sys.e1_states)
print(f'Excited subspace dimension: {len(sys.e1_states)}')

print('\n \n --------Interaction Terms------\n')
#print(sys.V_plus+sys.V_minus)

print('\n \n --------Effective Hamiltonian in gs------\n')
#print(sys.eff_hamiltonian_gs.simplify())
#print(sys.gs_states)