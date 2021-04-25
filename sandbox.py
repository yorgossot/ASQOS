import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import skeleton as ys

#Atoms. One in each cavity 
#Can support up to 5 atoms + 1 aux on a good pc

# x is aux 
# o is Borregaard atom
# - is fiber

system_string = 'o-x'

sys = ys.system(system_string)


print(f'System total dimension  {sys.dim}')

sys.construct_hamiltonian()

print(sys.hamiltonian)

sys.gs_hamiltonian()