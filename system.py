import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from components import *
from functions import *


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
                

        
    def construct_hamiltonian(self):
        '''
        Constructs Hamiltonian, corresponding state-vectors and corresponding excitation.
        '''
        #Hamiltonian construction
        H = zero_operator(self.flattened_dim_list)
        for elem in self.elements:
            H += elem.hamiltonian()
        self.hamiltonian = H

        #State and state excitation construction
        self.excitations = np.empty(self.dim, dtype= f'U{len(self.flattened_dim_list)}')
        self.states = np.empty(self.dim, dtype= f'U{len(self.flattened_dim_list) * 4 }')

        #This code does "tensor product" for characters
        times_to_be_tensored = 1
        sub_elem_dim_prod = 1
        for elem in self.elements:
            for sub_elem in elem.sub_elements:
                sub_elem_dim_prod *= sub_elem.dim
                for  t in range( times_to_be_tensored):
                    subblock_start = int(t/times_to_be_tensored*self.dim)
                    for i in range(sub_elem.dim):
                        slice_start = int( (i*self.dim / sub_elem_dim_prod ) + subblock_start  )
                        slice_end = int(( (i+1) *self.dim/sub_elem_dim_prod ) + subblock_start )
                        for j in range(slice_start,slice_end):
                            self.excitations[j] += sub_elem.excitations[i]
                            self.states[j] += sub_elem.states[i]
                times_to_be_tensored *= sub_elem.dim   
            
 

    def construct_gs_hamiltonian(self):
        '''
        Constructs ground state Hamiltonian, corresponding state-vectors and corresponding excitation.

        Note that the gs_hamiltonian will be a numpy array and not a qt objeect.
        '''
        
        self.gs_hamiltonian = np.copy(self.hamiltonian.full())      #ISSUE: Try to delete elements of the sparse matrix
        self.gs_states = np.copy(self.states)

        pos_to_del = []
        for (i , excitation ) in enumerate(self.excitations) :            
            for char in excitation:
                if char not in ('g' , 'q') : pos_to_del.append(i)

        self.gs_hamiltonian = np.delete(self.gs_hamiltonian , pos_to_del , axis= 0)
        self.gs_hamiltonian = np.delete(self.gs_hamiltonian , pos_to_del , axis= 1)
        self.gs_states = np.delete(self.gs_states, pos_to_del )


    def construct_e1_hamiltonian(self):
        '''
        Constructs the first excited state Hamiltonian, corresponding state-vectors and corresponding excitation.

        Note that the e1_hamiltonian will be a numpy array and not a qt objeect.
        '''
        
        self.e1_hamiltonian = np.copy(self.hamiltonian.full())      #ISSUE: Try to delete elements of the sparse matrix
        self.e1_states = np.copy(self.states)

        
        pos_to_del = []
        for (i , excitation ) in enumerate(self.excitations) : 
            e_num = 0
            f_num = 0
            for char in excitation:
                if char == 'e' or char == 'd' : 
                    e_num += 1 
                if char == 'q' : # we cant access starting state
                    pos_to_del.append(i) 
                    e_num += 2
            if e_num != 1 : pos_to_del.append(i) 


            

        self.e1_hamiltonian = np.delete(self.e1_hamiltonian , pos_to_del , axis= 0)
        self.e1_hamiltonian = np.delete(self.e1_hamiltonian , pos_to_del , axis= 1)
        self.e1_states = np.delete(self.e1_states, pos_to_del )
        self.e1_excitations = np.delete(self.excitations, pos_to_del )

        


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
            cavities_connected_pos = [dim_pos-2  , dim_pos+1]   
            self.sub_elements.append( fiber( dim_pos, cavities_connected_pos ))
        else:            
            print(f'Not valid element {type}. Give o , x and -')
            exit()

    def hamiltonian(self):
        H = zero_operator(self.system_dim_list)
        for sub_elem in self.sub_elements:
            H += sub_elem.hamiltonian()
        return H


        