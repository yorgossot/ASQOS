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
        
        self.H_coeffs = [sg.var("g", domain='real') ] *len(self.atom_dim_pos)
        self.gs_e1_interaction = [False] *len(self.atom_dim_pos)

        self.L_coeffs = [sg.sqrt( sg.var("kappa_c", domain='positive' ,  latex_name =r'\kappa_c')) ]


    def hamiltonian(self): 
        H = []
        for atom_dim_pos in self.atom_dim_pos:
            atom_dim = self.system_dim_list[atom_dim_pos]
            e_ket = qt.basis(atom_dim,2)
            f_ket = qt.basis(atom_dim,1)
            ef_ketbra = e_ket * f_ket.dag()

            tensor_list = id_operator_list(self.system_dim_list)
            tensor_list[self.dim_pos] = qt.destroy(self.dim)
            tensor_list[atom_dim_pos] = ef_ketbra

            H.append(qt.tensor(tensor_list) )
        return H 

    def lindblau(self):
        tensor_list = id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = qt.destroy(self.dim)

        l1 = [ qt.tensor(tensor_list) ]

        return l1 


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
        
        self.H_coeffs = [sg.var("v", domain='real' ,  latex_name =r'\nu'), sg.var("v")*sg.exp(sg.I*sg.var('phi', domain='real',  latex_name =r'\phi'))]
        self.gs_e1_interaction = [False,  False]

        self.L_coeffs = [sg.sqrt( sg.var("kappa_b", domain='positive' ,  latex_name =r'\kappa_b')) ]


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


    def lindblau(self):
        tensor_list = id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = qt.destroy(self.dim)

        l1 = [qt.tensor(tensor_list)]

        return l1 




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
        
        self.H_coeffs = [sg.var("De", domain='real' ,  latex_name =r'\Delta e') ]
        self.gs_e1_interaction = [False]

        self.L_coeffs = [sg.sqrt( sg.var("gamma", domain='positive' ,  latex_name =r'\gamma')) ]

    def hamiltonian(self):
        tensor_list = id_operator_list(self.system_dim_list)
        e_state_vector = qt.basis(self.dim,2)
        tensor_list[self.dim_pos] = e_state_vector.proj()
        H = [qt.tensor(tensor_list)]
        return H


    def lindblau(self):
        e_ket = qt.basis(self.dim,2)
        o_ket = qt.basis(self.dim,3)

        tensor_list = id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = o_ket*e_ket.dag()

        l1 = [qt.tensor(tensor_list)]

        return l1 



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
        self.laser_bool = True    
        
        self.H_coeffs = [sg.var("DE", domain='real' ,  latex_name =r'\Delta E') , sg.var("Omega", domain='real' , latex_name =r'\Omega')]
        self.gs_e1_interaction = [False,  self.laser_bool]     

        self.L_coeffs = [sg.sqrt( sg.var("gamma_g", domain='positive' ,  latex_name =r'\gamma_g')) , sg.sqrt( sg.var("gamma_f", domain='positive' ,  latex_name =r'\gamma_f'))]


    def hamiltonian(self):
        
        E_ket = qt.basis(self.dim,2)
        g_ket = qt.basis(self.dim,0)

        tensor_list = id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = E_ket.proj()
        H1 = qt.tensor(tensor_list)   #contrubution of excited level



        tensor_list = id_operator_list(self.system_dim_list)
        if self.laser_bool:
            tensor_list[self.dim_pos] = E_ket * g_ket.dag()
            H2 =  qt.tensor(tensor_list) 
        else:
            H2 = 0* qt.tensor(tensor_list) 
        
        return [H1 , H2]

    def lindblau(self):
        E_ket = qt.basis(self.dim,2)
        g_ket = qt.basis(self.dim,0)
        f_ket = qt.basis(self.dim,1)

        tensor_list = id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = f_ket*E_ket.dag()

        l1 = qt.tensor(tensor_list)


        tensor_list = id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = g_ket*E_ket.dag()

        l2 = qt.tensor(tensor_list)

        return [l1 , l2]

