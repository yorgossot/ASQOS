import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import skeleton as ys

#Atoms. One in each cavity

# x is aux 
# o is Borregaard atom
# - is fiber

system_string = 'x-o-o'

sys = ys.system(system_string)



print(sys.dim)

print(sys.hamiltonian())