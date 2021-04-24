import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

def construct_hamiltonian(system_string):
    '''
    Constructs Hamiltonian.

    Parameters:
    system_string: string
        Represents the system: O is aux atom, o are Borregaard atoms, - are fibers
    '''
    sys = system(system_string)
    print(sys.dim)
    for i in sys.elements:
        pass





class system:
    def __init__(self, system_string):
        self.size = len(system_string)
        self.elements = []
        self.dim = 1
        dim_pos = 0
        for ( pos , el_type ) in enumerate(system_string):
            self.elements.append( element( pos, el_type, dim_pos ) )
            dim_pos +=  self.elements[-1].size
            self.dim *=  self.elements[-1].dim
        


    def construct_hamiltonian():
        for el in elements:
            pass
    
            



class element:
    def __init__(self, pos , type, dim_pos):
        self.pos = pos
        self.type = type
        self.sub_elements = []
        if  type == 'O':
            self.size = 2
            self.dim = 2 * 3
            self.sub_elements.append( cavity(dim_pos) )
            dim_pos +=2
            self.sub_elements.append( qutrit(dim_pos) )
        elif type == 'o':            
            self.dim = 2 * 4
            self.size = 2
            self.sub_elements.append( cavity(dim_pos) )
            dim_pos+=2
            self.sub_elements.append( qunyb(dim_pos) )
        elif type == '-':
            self.size = 1
            self.dim = 2
            self.sub_elements.append( fiber( dim_pos))
        else:            
            print(f'Not valid element {type}')
            exit()
        
        
class cavity:
    def __init__(self,dim_pos):
        self.dim = 2
        self.dim_pos = dim_pos

class fiber:
    def __init__(self,dim_pos):
        self.dim = 2
        self.dim_pos = dim_pos

class qunyb:
    def __init__(self,dim_pos):
        self.dim = 4
        self.dim_pos = dim_pos

class qutrit:
    def __init__(self,dim_pos):
        self.dim=3
        self.dim_pos = dim_pos





