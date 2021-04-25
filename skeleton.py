import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

g = 1         #atom cavity coupling
delta_e = 1   #detuning of borregaard atom
delta_E = 1   #detuning of aux atom
Omega_2 = 1 #omega/2: rabi freq of laser on aux atom

class system:
    '''
    Class defining a system of elements.

    Initialize by giving a string containing:
    x : auxiliary atom
    o : Borregaard atom
    - : optical fiber

    ...

    Attributes
    ----------
    size : int
        Number of elements
    dim : int
        Total dimension of the system. 
    elements: list 
        List of all element objects.
    dim_list: list
        List with every dimension_list  of every element.
    flattened_dim_list : list
        List of all dimensions
    
    
    Methods
    -------
    hamiltonian
    '''
 
    def __init__(self, system_string):
        self.size = len(system_string)
        self.elements = []
        self.dim_list =[]
        self.dim = 1   
        dim_pos = 0
        for ( pos , el_type ) in enumerate(system_string):
            self.elements.append( element( pos, el_type, dim_pos ) )
            dim_pos +=  self.elements[-1].size
            self.dim *=  self.elements[-1].dim
            self.dim_list.append(self.elements[-1].dim_list)
        #Communicate the dimension_list to all (sub)elements
        flatten = lambda t: [item for sublist in t for item in sublist] #expression that flattens list
        flattened_list = flatten(self.dim_list)
        self.flattened_dim_list = flattened_list 
        for elem in self.elements:
            elem.system_dim_list = flattened_list
            for sub_elem in elem.sub_elements:
                sub_elem.system_dim_list = flattened_list
                

        
    def hamiltonian(self):
        H = zero_operator(self.flattened_dim_list)
        for elem in self.elements:
            H += elem.hamiltonian()
        return H 



class element:
    def __init__(self,  pos , type, dim_pos ):
        self.system_dim_list =[]
        self.pos = pos
        self.type = type
        self.dim_pos = dim_pos
        self.sub_elements = []
        if  type == 'x':
            self.size = 2
            self.dim = 2 * 3
            self.dim_list = [2 , 3]
            cavity_dim_pos = dim_pos
            atom_dim_pos = cavity_dim_pos + 1
            self.sub_elements.append( cavity(cavity_dim_pos,atom_dim_pos) )
            self.sub_elements.append( qutrit(atom_dim_pos , cavity_dim_pos ) )
        elif type == 'o':            
            self.dim = 2 * 4
            self.dim_list = [2 , 4]
            self.size = 2
            cavity_dim_pos = dim_pos
            atom_dim_pos = cavity_dim_pos + 1
            self.sub_elements.append( cavity( cavity_dim_pos, atom_dim_pos) )
            self.sub_elements.append( qunyb(atom_dim_pos , cavity_dim_pos ) )
        elif type == '-':
            self.size = 1
            self.dim = 2
            self.dim_list = [2]
            self.sub_elements.append( fiber( dim_pos))
        else:            
            print(f'Not valid element {type}')
            exit()

    def hamiltonian(self):
        H = zero_operator(self.system_dim_list)
        for sub_elem in self.sub_elements:
            H += sub_elem.hamiltonian()
        return H


        
class cavity:
    def __init__(self , dim_pos , atom_dim_pos  ):
        self.dim = 2
        self.dim_pos = dim_pos
        self.atom_dim_pos = atom_dim_pos
        self.system_dim_list =[]
    
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
    def __init__(self,dim_pos):
        self.dim = 2
        self.dim_pos = dim_pos
        self.system_dim_list =[]

    def hamiltonian(self): #return 0 operator
        H = zero_operator(self.system_dim_list)
        return H 

class qunyb:
    '''
    Borregaard atom with 4 levels.

    index | state
       0  |   0
       1  |   1
       e  |   2
       o  |   3

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

    def hamiltonian(self):
        tensor_list = id_operator_list(self.system_dim_list)
        e_state_vector = qt.basis(self.dim,2)
        tensor_list[self.dim_pos] = e_state_vector.proj()
        return delta_e * qt.tensor(tensor_list)


class qutrit:
    '''
    Auxiliary atom with 3 levels.
    Laser drives |g><e|
    
    index | state
       g  |   0
       f  |   1
       E  |   2
    '''
    def __init__(self,dim_pos,cavity_dim_pos):
        self.dim=3
        self.dim_pos = dim_pos
        self.system_dim_list =[]
        self.cavity_dim_pos = cavity_dim_pos
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



def zero_operator(dim_list):
    '''
    Returns an operator with zero entries for a given dimension list.
    '''
    tensor_list = []
    for i in dim_list:
        tensor_list = qt.identity(dim_list) 
    return 0*qt.tensor(tensor_list)

def id_operator(dim_list):
    '''
    Returns an identity operator for a given dimension list.
    '''
    tensor_list = []
    for i in dim_list:
        tensor_list = qt.identity(dim_list) 
    return qt.tensor(tensor_list)

def id_operator_list(dim_list):
    '''
    Returns an identity operator list for a given dimension list. Used for tensor products.
    '''
    tensor_list = []
    for  dim in dim_list:
        tensor_list.append(qt.identity(dim))   
    return tensor_list