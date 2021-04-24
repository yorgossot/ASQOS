import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import skeleton as ys

#Atoms. One in each cavity

# O is aux
system_string = 'O'

sys = ys.system(system_string)

print(sys.elements[0].system_dim_list)

print(sys.dim)

print(sys.hamiltonian())