import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from functions import *

g = 1         #atom cavity coupling
delta_e = 1   #detuning of borregaard atom
delta_E = 1   #detuning of aux atom
Omega_2 = 1   #omega/2: rabi freq of laser on aux atom
#Fiber
v_fiber = 1   #cavity fiber coupling
phi     = 0   #phase in the hamiltonian

class cavity:
    '''
    Cavity with two levels containing a single atom. 

    index | state  |  excitation
       0  |   0    |     g
       1  |   1    |     e

    ...

    Attributes
    ----------

    Methods
    -------
   
    '''    
    def __init__(self , dim_pos , atom_dim_pos  ):
        self.dim = 2
        self.dim_pos = dim_pos
        self.atom_dim_pos = atom_dim_pos
        self.system_dim_list =[]
        self.excitations = np.array(['g','e'])
        self.states = np.array(['0','1']) 

    def hamiltonian(self): 
        atom_dim = self.system_dim_list[self.atom_dim_pos]
        e_ket = qt.basis(atom_dim,2)
        f_ket = qt.basis(atom_dim,1)
        ef_ketbra = e_ket * f_ket.dag()

        tensor_list = id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = qt.destroy(self.dim)
        tensor_list[self.atom_dim_pos] = ef_ketbra

        H = g*qt.tensor(tensor_list)
        H += H.dag()
        return H 

class fiber:
    '''
    Fiber coonecting two cavities.

    index | state  |  excitation
       0  |   0    |     g
       1  |   1    |     e

    ...

    Attributes
    ----------

    Methods
    -------
   
    '''
    def __init__(self,dim_pos, cavities_connected_pos ):
        self.dim = 2
        self.dim_pos = dim_pos
        self.cavities_connected_pos = cavities_connected_pos
        self.system_dim_list =[]
        self.excitations = np.array(['g','e'])
        self.states = np.array(['0','1']) 

    def hamiltonian(self): #return 0 operator
        H = zero_operator(self.system_dim_list)
        
        #adds all contributions that destroy photons in the fiber
        tensor_list = id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] =  qt.destroy(self.dim)        
        for (i,cavity_pos) in enumerate(self.cavities_connected_pos):
            cavity_dim = self.system_dim_list[cavity_pos]
            tensor_list[cavity_pos] = qt.create(cavity_dim)
            phase = phi
            if phase != 0 and i!=0 :
                H += qt.tensor(tensor_list) * np.exp( complex(0,phase) )
            else:
                H += qt.tensor(tensor_list)
            tensor_list = id_operator_list(self.system_dim_list)
            tensor_list[self.dim_pos] =  qt.destroy(self.dim) 
        
        H = v_fiber * H
        H += H.dag()     #add hermitian conj
        
        return H 

class qunyb:
    '''
    Borregaard atom with 4 levels.

    index | state |  excitation 
       0  |   0   |      g
       1  |   1   |      g
       e  |   2   |      e
       o  |   3   |      d

    ...

    Attributes
    ----------

    Methods
    -------
   
    '''
    def __init__(self,dim_pos,cavity_dim_pos):
        self.dim = 4
        self.dim_pos = dim_pos
        self.system_dim_list =[]
        self.cavity_dim_pos = cavity_dim_pos    
        self.excitations = np.array(['g', 'g', 'e' , 'd'])
        self.states = np.array(['0','1' , 'e' , 'o']) 

    def hamiltonian(self):
        tensor_list = id_operator_list(self.system_dim_list)
        e_state_vector = qt.basis(self.dim,2)
        tensor_list[self.dim_pos] = e_state_vector.proj()
        return delta_e * qt.tensor(tensor_list)


class qutrit:
    '''
    Auxiliary atom with 3 levels.
    Laser drives |g><e|
    
    index | state |  excitation
       g  |   0   |      q
       f  |   1   |      p
       E  |   2   |      e
    '''
    def __init__(self,dim_pos,cavity_dim_pos):
        self.dim=3
        self.dim_pos = dim_pos
        self.system_dim_list =[]
        self.cavity_dim_pos = cavity_dim_pos
        self.excitations = np.array(['q', 'p', 'e' ])
        self.states = np.array(['g','f' , 'E']) 
        self.laser_bool = True    #laser interacts

    def hamiltonian(self):
        E_ket = qt.basis(self.dim,2)
        g_ket = qt.basis(self.dim,0)

        tensor_list = id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = E_ket.proj()
        H = delta_E * qt.tensor(tensor_list)   #contrubution of excited level

        if self.laser_bool:
            tensor_list = id_operator_list(self.system_dim_list)
            tensor_list[self.dim_pos] = E_ket * g_ket.dag()
            laser_cont = Omega_2 * qt.tensor(tensor_list) 
            
            H += laser_cont + laser_cont.dag()   #contrubution of laser drive

        return H