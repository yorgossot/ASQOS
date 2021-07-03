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
            self.sub_elements.append( cavity(cavity_dim_pos) )  
            self.sub_elements.append( qutrit(atom_dim_pos , cavity_dim_pos ) )
        elif type == 'o':            
            self.dim = 2 * 4
            self.dim_list = [2 , 4]
            self.size = 2
            cavity_dim_pos = dim_pos
            atom_dim_pos = cavity_dim_pos + 1
            self.sub_elements.append( cavity( cavity_dim_pos) ) 
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
            self.size = 5
            self.dim = 2 *3 * 4 * 4 * 4
            self.dim_list = [2 , 3 , 4 , 4 , 4] 
            cavity_dim_pos = dim_pos
            atom_dim_pos = [cavity_dim_pos + 1   , cavity_dim_pos + 2 , cavity_dim_pos + 3 ,  cavity_dim_pos + 4]
            self.sub_elements.append( cavity(cavity_dim_pos ))
            self.sub_elements.append( qutrit(atom_dim_pos[0], cavity_dim_pos ) )
            self.sub_elements.append( ququad(atom_dim_pos[1] , cavity_dim_pos ) )
            self.sub_elements.append( ququad(atom_dim_pos[2] , cavity_dim_pos ) )
            self.sub_elements.append( ququad(atom_dim_pos[3] , cavity_dim_pos ) )
        elif type == '1':
            #borregaard 2015 with 1+1 atoms
            self.size = 3
            self.dim = 2 *3 * 4 
            self.dim_list = [2 , 3  , 4] 
            cavity_dim_pos = dim_pos
            atom_dim_pos = [cavity_dim_pos + 1   , cavity_dim_pos + 2 ]
            self.sub_elements.append( cavity(cavity_dim_pos ))
            self.sub_elements.append( qutrit(atom_dim_pos[0], cavity_dim_pos ) )
            self.sub_elements.append( ququad(atom_dim_pos[1] , cavity_dim_pos ) )
        elif type == 'T':
            #T symmetric shaped configuration
            self.size = 7
            self.dim = 2*3 * 2*4*2  * 2*4*2 * 2*4*2
            self.dim_list = [2,3 , 2,4,2  , 2,4,2 , 2,4,2]
            aux_cavity_dim_pos = dim_pos
            aux_atom_dim_pos = dim_pos+1
            self.sub_elements.append( cavity(aux_cavity_dim_pos) ) 
            self.sub_elements.append( qutrit(aux_atom_dim_pos, aux_cavity_dim_pos ) )
            for i in range(3):
                o_cavity_dim_pos = 3*i + 2
                o_atom_dim_pos   = 3*i + 3
                fiber_dim_pos    = 3*i + 4
                fiber_connected_cavities = [o_cavity_dim_pos , aux_cavity_dim_pos]
                self.sub_elements.append( cavity(o_cavity_dim_pos) ) 
                self.sub_elements.append( ququad(o_atom_dim_pos , o_cavity_dim_pos ) )
                self.sub_elements.append( fiber( fiber_dim_pos, fiber_connected_cavities ))
        elif type == 't':
            #t semi symmetric shaped configuration
            self.size = 7
            self.dim = 2*3 * 2*4*2  * 2*4*2 * 2*4*2
            self.dim_list = [2,3 , 2,4,2  , 2,4,2 , 2,4,2]
            aux_cavity_dim_pos = dim_pos
            aux_atom_dim_pos = dim_pos+1
            self.sub_elements.append( cavity(aux_cavity_dim_pos) ) 
            self.sub_elements.append( qutrit(aux_atom_dim_pos, aux_cavity_dim_pos ) )
            
            ox_cavity_dim_pos =  2
            o_atom_dim_pos   =  3
            fiber_dim_pos    =  4
            fiber_connected_cavities = [ox_cavity_dim_pos , aux_cavity_dim_pos]
            self.sub_elements.append( cavity(ox_cavity_dim_pos) ) 
            self.sub_elements.append( ququad(o_atom_dim_pos , ox_cavity_dim_pos ) )
            self.sub_elements.append( fiber( fiber_dim_pos, fiber_connected_cavities ))
            for i in range(2):
                o_cavity_dim_pos = 3*i + 5
                o_atom_dim_pos   = 3*i + 6
                fiber_dim_pos    = 3*i + 7
                fiber_connected_cavities = [o_cavity_dim_pos , ox_cavity_dim_pos]
                self.sub_elements.append( cavity(o_cavity_dim_pos) ) 
                self.sub_elements.append( ququad(o_atom_dim_pos , o_cavity_dim_pos ) )
                self.sub_elements.append( fiber( fiber_dim_pos, fiber_connected_cavities ))
        else:            
            print(f'Not valid element {type}. Give o , x, - or other valid configuration')
            exit()