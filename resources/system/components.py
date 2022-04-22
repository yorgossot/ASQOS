#
# File containing all fundamental components of the system and element class.
#

import qutip as qt
import numpy as np
import sage.all as sg
from . import system_functions

# example of component class
'''
class component_name:  
    def __init__(self , dim_pos  ):
        self.dim : dimension of subspace describing the component
        self.dim_pos : position of the dimensions in the whole system
        self.system_dim_list : list of dimensions in the whole system. It is set from the system class.
        self.excitations : numpy array with characters corresponding to the excitations of each level of the system 
            'g' -> ground state
            'e' -> 1st excited state
            'd' -> decayed state
            'q' -> ground state of auxiliary atom g
            'p' -> meta stable state of auxialiary atom f
        
        self.states : numpy array with characters corresponding to the name of each level of the system 
        
        # Let there be n Hamiltonian interactions
        self.H_coeffs : list of n coefficients of the Hamiltonian interactions (sagemath variables)
        self.gs_e1_interaction : list of n booleans denoting whether it is an excitation interaction
        self.TwoPhotonResonance = True

        # Let there be m loss operators
        self.L_coeffs :  list of m coefficients of the Lindblad operators (sagemath variables)

    def update_index(self, variable_index):
        return variable_index

    def hamiltonian(self): 
        Returns a list of qt objects that contain the ketbra of the interaction.
        There has to be correspondence between this list and self.H_coeffs .
        Because the hamiltonian is hermitian, the conjugate is potentially added by the system class.

    def lindblad(self):
        Returns a list of qt objects that contain the ketbra of the loss.
        There has to be correspondence between this list and self.L_coeffs .
'''

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
    def __init__(self , dim_pos  ):
        self.dim = 2
        self.dim_pos = dim_pos
        self.system_dim_list =[]
        self.excitations = np.array(['g','e'])
        self.states = np.array(['0','1']) 
        
        self.H_coeffs = [] 
        self.gs_e1_interaction = [] 
        self.TwoPhotonResonance = True

        self.L_coeffs = [sg.sqrt( sg.var("kappa_c", domain='positive' ,  latex_name =r'\kappa_c')) ]

    def update_index(self, variable_index):
        return variable_index

    

    def hamiltonian(self): 
        H = []

        if self.TwoPhotonResonance == False:
            self.H_coeffs.append( sg.var("dc", domain='positive' ,  latex_name =fr'{{\delta }}_{{c}}'))
            self.gs_e1_interaction.append(False)
            
            tensor_list = system_functions.id_operator_list(self.system_dim_list)
            e_state_vector = qt.basis(self.dim,1)
            tensor_list[self.dim_pos] = e_state_vector.proj()
            H.append(qt.tensor(tensor_list))

        return H 

    def lindblad(self):
        tensor_list = system_functions.id_operator_list(self.system_dim_list)
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
        
        self.H_coeffs = [sg.var("v", domain='positive' ,  latex_name =r'\nu'), sg.var("v")*sg.exp(sg.I*sg.var('phi', domain='positive',  latex_name =r'\phi'))]
        self.gs_e1_interaction = [False,  False]
        self.TwoPhotonResonance = True

        self.L_coeffs = [sg.sqrt( sg.var("kappa_b", domain='positive' ,  latex_name =r'\kappa_b')) ]

    def update_index(self, variable_index):
        return variable_index

    def hamiltonian(self): #return 0 operator
        H = system_functions.zero_operator(self.system_dim_list)
        
        
        #if fiber in the end, interacts with the first cavity
        if self.cavities_connected_pos[-1]>len(self.system_dim_list)-1:
            self.cavities_connected_pos[-1] = 0
        


        #adds all contributions that destroy photons in the fiber
        

        H = []
        for (i,cavity_pos) in enumerate(self.cavities_connected_pos):
            tensor_list = system_functions.id_operator_list(self.system_dim_list)
            tensor_list[self.dim_pos] =  qt.destroy(self.dim)  
            cavity_dim = self.system_dim_list[cavity_pos]
            tensor_list[cavity_pos] = qt.create(cavity_dim)
            H.append( qt.tensor(tensor_list)  )


        if self.TwoPhotonResonance == False:
            self.H_coeffs.append( sg.var("db", domain='positive' ,  latex_name =fr'{{\delta }}_{{b}}') )
            self.gs_e1_interaction .append( False )
            
            tensor_list = system_functions.id_operator_list(self.system_dim_list)
            e_state_vector = qt.basis(self.dim,1)
            tensor_list[self.dim_pos] = e_state_vector.proj()
            H.append(qt.tensor(tensor_list))

        return H 


    def lindblad(self):
        tensor_list = system_functions.id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = qt.destroy(self.dim)

        l1 = [qt.tensor(tensor_list)]

        return l1 




class ququad:
    '''
    Atom with 4 levels.

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
        
        self.H_coeffs = [sg.var("De", domain='positive' ,  latex_name =r'\Delta e') , sg.var("g", domain='positive')]
        self.gs_e1_interaction = [False , False]
        self.TwoPhotonResonance = True

        self.L_coeffs = [sg.sqrt( sg.var("gamma", domain='positive' ,  latex_name =r'\gamma')) ]

    def update_index(self, variable_index):
        self.H_coeffs = [sg.var(f'De{variable_index}', domain='positive' ,  latex_name =fr'{{\Delta e}}_{{{variable_index}}}') , sg.var("g", domain='positive')]
        return variable_index + 1
    
    def hamiltonian(self):
        H = []
        
        tensor_list = system_functions.id_operator_list(self.system_dim_list)
        e_state_vector = qt.basis(self.dim,2)
        tensor_list[self.dim_pos] = e_state_vector.proj()
        H.append(qt.tensor(tensor_list))


        #atom cavity interaction
        e_ket = qt.basis(self.dim,2)
        f_ket = qt.basis(self.dim,1)
        ef_ketbra = e_ket * f_ket.dag()
        
        cavity_dim = self.system_dim_list[self.cavity_dim_pos]
        
        tensor_list = system_functions.id_operator_list(self.system_dim_list)
        tensor_list[self.cavity_dim_pos] = qt.destroy(cavity_dim)
        tensor_list[self.dim_pos] = ef_ketbra

        H.append(qt.tensor(tensor_list) )

        return H


    def lindblad(self):
        e_ket = qt.basis(self.dim,2)
        o_ket = qt.basis(self.dim,3)

        tensor_list = system_functions.id_operator_list(self.system_dim_list)
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
        
        self.H_coeffs = [sg.var("DE", domain='positive' ,  latex_name =r'\Delta E') , sg.var("Omega", domain='positive' , latex_name =r'\Omega') , sg.var("g_f", domain='positive', latex_name =r'g_f')]
        self.gs_e1_interaction = [False,  self.laser_bool  , False]     
        self.TwoPhotonResonance = True

        self.L_coeffs = [ sg.sqrt( sg.var("gamma_f", domain='positive' ,  latex_name =r'\gamma_f')), sg.sqrt( sg.var("gamma_g", domain='positive' ,  latex_name =r'\gamma_g')) ]

    def update_index(self, variable_index):
        return variable_index

    def hamiltonian(self):
        
        E_ket = qt.basis(self.dim,2)
        g_ket = qt.basis(self.dim,0)

        tensor_list = system_functions.id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = E_ket.proj()
        H1 = qt.tensor(tensor_list)   #contrubution of excited level


        #laser interaction
        tensor_list = system_functions.id_operator_list(self.system_dim_list)
        if self.laser_bool:
            tensor_list[self.dim_pos] = E_ket * g_ket.dag()
            H2 =  qt.tensor(tensor_list) 
        else:
            H2 = 0* qt.tensor(tensor_list) 
        

        #atom cavity interaction
        e_ket = qt.basis(self.dim,2)
        f_ket = qt.basis(self.dim,1)
        ef_ketbra = e_ket * f_ket.dag()
        
        cavity_dim = self.system_dim_list[self.cavity_dim_pos]
        
        tensor_list = system_functions.id_operator_list(self.system_dim_list)
        tensor_list[self.cavity_dim_pos] = qt.destroy(cavity_dim)
        tensor_list[self.dim_pos] = ef_ketbra

        H3 = qt.tensor(tensor_list) 

        H =  [H1 , H2 , H3]


        return H

    def lindblad(self):
        E_ket = qt.basis(self.dim,2)
        g_ket = qt.basis(self.dim,0)
        f_ket = qt.basis(self.dim,1)

        tensor_list = system_functions.id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = f_ket*E_ket.dag()

        l1 = qt.tensor(tensor_list)


        tensor_list = system_functions.id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = g_ket*E_ket.dag()

        l2 = qt.tensor(tensor_list)

        return [l1 , l2]




class quhex:
    '''
    Atom with 5 levels.

    index | state |  excitation 
       0  |   0   |      g
       1  |   1   |      g
       e  |   2   |      e
       o  |   3   |      d
       X  |   4   |      e   (excited coupled to 0 state)
       O  |   5   |      d
    ...

    Attributes
    ----------

    Methods
    -------
   
    '''
    def __init__(self,dim_pos,cavity_dim_pos):
        self.dim = 6
        self.dim_pos = dim_pos
        self.system_dim_list =[]
        self.cavity_dim_pos = cavity_dim_pos
        self.excitations = np.array(['g', 'g', 'e' , 'd','e','d'])
        self.states = np.array(['0','1' , 'e' , 'o', 'X','O']) 
        
        self.H_coeffs = [sg.var("De", domain='positive' ,  latex_name =r'\Delta e') ,sg.var("De0", domain='positive' ,  latex_name =r'\Delta e_0'), \
             sg.var("g", domain='positive') , sg.var("g0", domain='positive',  latex_name =r'g_0') ]
        self.gs_e1_interaction = [False , False , False , False]
        self.TwoPhotonResonance = True

        self.L_coeffs = [sg.sqrt( sg.var("gamma", domain='positive' ,  latex_name =r'\gamma')) ,sg.sqrt( sg.var("gamma0", domain='positive' ,  latex_name =r'\gamma_0')) ]

    def update_index(self, variable_index):
        self.H_coeffs = [sg.var(f"De{variable_index}", domain='positive' ,  latex_name =r'\Delta e') ,sg.var(f"De0{variable_index}", domain='positive' ,  latex_name =r'\Delta e_0'), \
             sg.var("g", domain='positive') , sg.var("g0", domain='positive',  latex_name =r'g_0') ]
        return variable_index + 1
    
    def hamiltonian(self):
        H = []
        
        #e detuning
        tensor_list = system_functions.id_operator_list(self.system_dim_list)
        e_state_vector = qt.basis(self.dim,2)
        tensor_list[self.dim_pos] = e_state_vector.proj()
        H.append(qt.tensor(tensor_list))

        #X detuning
        tensor_list = system_functions.id_operator_list(self.system_dim_list)
        X_state_vector = qt.basis(self.dim,4)
        tensor_list[self.dim_pos] = X_state_vector.proj()
        H.append(qt.tensor(tensor_list))

        #atom cavity interaction with state 1
        e_ket = qt.basis(self.dim,2)
        f_ket = qt.basis(self.dim,1)
        ef_ketbra = e_ket * f_ket.dag()
        
        cavity_dim = self.system_dim_list[self.cavity_dim_pos]
        
        tensor_list = system_functions.id_operator_list(self.system_dim_list)
        tensor_list[self.cavity_dim_pos] = qt.destroy(cavity_dim)
        tensor_list[self.dim_pos] = ef_ketbra

        H.append(qt.tensor(tensor_list) )

        #atom cavity interaction with state 0
        X_ket = qt.basis(self.dim,4)
        f0_ket = qt.basis(self.dim,0)
        Xf0_ketbra = X_ket * f0_ket.dag()
        
        cavity_dim = self.system_dim_list[self.cavity_dim_pos]
        
        tensor_list = system_functions.id_operator_list(self.system_dim_list)
        tensor_list[self.cavity_dim_pos] = qt.destroy(cavity_dim)
        tensor_list[self.dim_pos] = Xf0_ketbra

        H.append(qt.tensor(tensor_list) )
        return H

    def lindblad(self):
        l1 = []

        e_ket = qt.basis(self.dim,2)
        o_ket = qt.basis(self.dim,3)

        tensor_list = system_functions.id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = o_ket*e_ket.dag()

        l1.append(qt.tensor(tensor_list))

        X_ket = qt.basis(self.dim,4)
        o_ket = qt.basis(self.dim,5)

        tensor_list = system_functions.id_operator_list(self.system_dim_list)
        tensor_list[self.dim_pos] = o_ket*X_ket.dag()

        l1.append(qt.tensor(tensor_list))

        return l1 