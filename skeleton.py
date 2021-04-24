import qutip as qt
import numpy as np
import matplotlib.pyplot as plt


class system:
    
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

        flatten = lambda t: [item for sublist in t for item in sublist] #expression that flattens list
        flattened_list = flatten(self.dim_list)
        for elem in self.elements:
            elem.system_dim_list = flattened_list
            for sub_elem in elem.sub_elements:
                sub_elem.system_dim_list = flattened_list
                

        
    def hamiltonian(self):
        H = np.zeros((self.dim,self.dim))
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
        if  type == 'O':
            self.size = 2
            self.dim = 2 * 3
            self.dim_list = [2 , 3]
            cavity_dim_pos = dim_pos
            atom_dim_pos = cavity_dim_pos + 2
            self.sub_elements.append( cavity(cavity_dim_pos) )
            self.sub_elements.append( qutrit(atom_dim_pos) )
        elif type == 'o':            
            self.dim = 2 * 4
            self.dim_list = [2 , 4]
            self.size = 2
            cavity_dim_pos = dim_pos
            atom_dim_pos = cavity_dim_pos + 2
            self.sub_elements.append( cavity(cavity_dim_pos) )
            self.sub_elements.append( qunyb(atom_dim_pos) )
        elif type == '-':
            self.size = 1
            self.dim = 2
            self.dim_list = [2]
            self.sub_elements.append( fiber( dim_pos))
        else:            
            print(f'Not valid element {type}')
            exit()

    def hamiltonian(self):
        H = 0
        for sub_elem in self.sub_elements:
            H += sub_elem.hamiltonian()


        
class cavity:
    def __init__(self , dim_pos  ):
        self.dim = 2
        self.dim_pos = dim_pos
        self.system_dim_list =[]
    
    def hamiltonian(self):
        pass

class fiber:
    def __init__(self,dim_pos):
        self.dim = 2
        self.dim_pos = dim_pos
        self.system_dim_list =[]

class qunyb:
    def __init__(self,dim_pos):
        self.dim = 4
        self.dim_pos = dim_pos
        self.system_dim_list =[]
    

class qutrit:
    def __init__(self,dim_pos):
        self.dim=3
        self.dim_pos = dim_pos
        self.system_dim_list =[]





