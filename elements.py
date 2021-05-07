import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import sage.all as sg
from collections import Counter
import time

from components import *
from functions import *

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
            self.sub_elements.append( cavity(cavity_dim_pos,[atom_dim_pos]) )  #atom_dim_pos given as a list to cavity
            self.sub_elements.append( qutrit(atom_dim_pos , cavity_dim_pos ) )
        elif type == 'o':            
            self.dim = 2 * 4
            self.dim_list = [2 , 4]
            self.size = 2
            cavity_dim_pos = dim_pos
            atom_dim_pos = cavity_dim_pos + 1
            self.sub_elements.append( cavity( cavity_dim_pos, [atom_dim_pos]) ) #atom_dim_pos given as a list to cavity
            self.sub_elements.append( ququad(atom_dim_pos , cavity_dim_pos ) )
        elif type == '-':
            self.size = 1
            self.dim = 2
            self.dim_list = [2] 
            cavities_connected_pos = [dim_pos-2  , dim_pos+1]   
            self.sub_elements.append( fiber( dim_pos, cavities_connected_pos ))
        elif type == '2':
            #borregaard 2015 with 1+2 atoms
            self.size = 4
            self.dim = 2 *3 * 4 * 4
            self.dim_list = [2 , 3 , 4 , 4] 
            cavity_dim_pos = dim_pos
            atom_dim_pos = [cavity_dim_pos + 1   , cavity_dim_pos + 2 , cavity_dim_pos + 3]
            self.sub_elements.append( cavity(cavity_dim_pos ))
            self.sub_elements.append( qutrit(atom_dim_pos[0], cavity_dim_pos ) )
            self.sub_elements.append( ququad(atom_dim_pos[1] , cavity_dim_pos ) )
            self.sub_elements.append( ququad(atom_dim_pos[2] , cavity_dim_pos ) )
        elif type == '3':
            #borregaard 2015 with 1+3 atoms
            self.size = 4
            self.dim = 2 *3 * 4 * 4 * 4
            self.dim_list = [2 , 3 , 4 , 4 , 4] 
            cavity_dim_pos = dim_pos
            atom_dim_pos = [cavity_dim_pos + 1   , cavity_dim_pos + 2 , cavity_dim_pos + 3 ,  cavity_dim_pos + 4]
            self.sub_elements.append( cavity(cavity_dim_pos ))
            self.sub_elements.append( qutrit(atom_dim_pos[0], cavity_dim_pos ) )
            self.sub_elements.append( ququad(atom_dim_pos[1] , cavity_dim_pos ) )
            self.sub_elements.append( ququad(atom_dim_pos[2] , cavity_dim_pos ) )
            self.sub_elements.append( ququad(atom_dim_pos[3] , cavity_dim_pos ) )
        else:            
            print(f'Not valid element {type}. Give o , x and -')
            exit()