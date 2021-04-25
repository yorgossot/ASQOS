import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

g = 1         #atom cavity coupling
delta_e = 1   #detuning of borregaard atom
delta_E = 1   #detuning of aux atom
Omega_2 = 1   #omega/2: rabi freq of laser on aux atom
#Fiber
v_fiber = 1   #cavity fiber coupling
phi     = 0   #phase in the hamiltonian

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