import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
import sage.all as sg
from functions import *

#g = 1         #atom cavity coupling
#delta_e = 0   #detuning of borregaard atom
#delta_E = 0   #detuning of aux atom
#Omega_2 = 0   #omega/2: rabi freq of laser on aux atom
#Fiber
#v_fiber = 0   #cavity fiber coupling
#phi     = 0   #phase in the hamiltonian

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
        self.H_coeffs = [sg.var("g") ]

    def hamiltonian(self): 
        atom_dim = self.system_dim_list[self.atom_dim_pos]
        e_ket = qt.basis(atom_dim,2)
        f_ket = qt.basis(atom_dim,1)
        ef_ketbra = e_ket * f_ket.dag()

        tensor_list = id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = qt.destroy(self.dim)
        tensor_list[self.atom_dim_pos] = ef_ketbra

        H = [qt.tensor(tensor_list)]
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
        self.H_coeffs = [sg.var("v"), sg.var("v")*np.exp(complex(0,1)*sg.var('phi'))]

    def hamiltonian(self): #return 0 operator
        H = zero_operator(self.system_dim_list)
        
        
        #if fiber in the end, interacts with the first cavity
        if self.cavities_connected_pos[-1]>len(self.system_dim_list)-1:
            self.cavities_connected_pos[-1] = 0
        


        #adds all contributions that destroy photons in the fiber
        

        H = []
        for (i,cavity_pos) in enumerate(self.cavities_connected_pos):
            tensor_list = id_operator_list(self.system_dim_list)
            tensor_list[self.dim_pos] =  qt.destroy(self.dim)  
            cavity_dim = self.system_dim_list[cavity_pos]
            tensor_list[cavity_pos] = qt.create(cavity_dim)
            H.append( qt.tensor(tensor_list)  )

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
        self.H_coeffs = [sg.var("De") ]

    def hamiltonian(self):
        tensor_list = id_operator_list(self.system_dim_list)
        e_state_vector = qt.basis(self.dim,2)
        tensor_list[self.dim_pos] = e_state_vector.proj()
        H = [qt.tensor(tensor_list)]
        return H


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
        self.H_coeffs = [sg.var("DE") , sg.var("Omega")]

    def hamiltonian(self):
        
        E_ket = qt.basis(self.dim,2)
        g_ket = qt.basis(self.dim,0)

        tensor_list = id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = E_ket.proj()
        H1 = qt.tensor(tensor_list)   #contrubution of excited level



        tensor_list = id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = E_ket * g_ket.dag()
        H2 =  qt.tensor(tensor_list) 
        
        return [H1 , H2]