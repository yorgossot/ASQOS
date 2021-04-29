import qutip as qt
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import sage.all as sg
from collections import Counter
import time

from components import *
from functions import *

class system:
    '''
    Class defining a system of elements.

    Initialize by giving a string containing:
    x : auxiliary atom
    o : Borregaard atom
    - : optical fiber
    
    Corresponding state-vectors and corresponding excitations will be initialized.

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

        self.update_subelements()
        print('Constructing states and excitations...')
        self.construct_states_and_excitations()
        print('Constructing ground and first-excited statespace...')
        self.construct_gs_e1_dec_subspace()
        self.obtain_energy_info()    
        print('Constructing gs_hamiltonian ...')  
        self.construct_gs_hamiltonian()
        print('Constructing e1_hamiltonian ...') 
        self.construct_e1_hamiltonian()
        print('Constructing interactions V_plus and V_minus ...')
        self.construct_V()
        print('Constructing NJ_hamiltonian ...')        
        self.construct_nj_hamiltonian()
        self.construct_nj_hamiltonian_inverse()
        print('Constructing eff_hamiltonian and effective lindblau operators ...')   
        self.construct_eff_hamiltonian_lindblaus()
        print('Solving effective Lindblau master equation ...') 
        self.solve_master_equation()
        print(f'\nSystem  {system_string}  initialized!')


    def update_subelements(self):
        '''
        Communicate the dimension_list to all (sub)elements
        '''
        flatten = lambda t: [item for sublist in t for item in sublist] #expression that flattens list
        flattened_list = flatten(self.dim_list)
        self.flattened_dim_list = flattened_list 
        for elem in self.elements:
            elem.system_dim_list = flattened_list
            for sub_elem in elem.sub_elements:
                sub_elem.system_dim_list = flattened_list



    def construct_states_and_excitations(self):
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


                
    def construct_gs_e1_dec_subspace(self):
        
        self.pos_to_del_gs_e1_dec = []
        for (i , excitation ) in enumerate(self.excitations) :
            letter_count = Counter(excitation)
            g_count = letter_count['g'] #0/1 states
            q_count = letter_count['q'] #g states
            e_count = letter_count['e'] #e/E states
            p_count = letter_count['p'] #f states
            d_count = letter_count['d'] #o states
          
            gs_del_flag = False
            if e_count!=0 or p_count!=0 or d_count!=0: gs_del_flag = True  #no excitation/no f state/no decay/
            e1_del_flag = False
            if e_count!=1 or q_count!=0 or d_count!=0: e1_del_flag = True  #1 exc /no g state/no decay  
            dec_del_flag = False
            if e_count!=0 or q_count!=0 or d_count>1: dec_del_flag = True  #0 exc /no g state/ 1 or 0 decay  

            if e1_del_flag and gs_del_flag and dec_del_flag:
                self.pos_to_del_gs_e1_dec.append(i)
        
        #self.pos_gs_e1_dec = [i for i in [*range(self.dim)] if i not in self.pos_to_del_gs_e1_dec]   DELETED DUE TO IT BEING TOO SLOW
        self.gs_e1_dec_dim = self.dim - len(self.pos_to_del_gs_e1_dec)
        self.gs_e1_dec_excitations = np.delete(self.excitations , self.pos_to_del_gs_e1_dec)
        self.gs_e1_dec_states = np.delete(self.states , self.pos_to_del_gs_e1_dec)


        # gs_hamiltonian states and positions in gs_e1_dec subspace

        self.pos_to_del_gs = []
        for (i , excitation ) in enumerate(self.gs_e1_dec_excitations) :            
            letter_count = Counter(excitation)
            e_count = letter_count['e'] #e/E states
            p_count = letter_count['p'] #f states
            d_count = letter_count['d'] #o states
          
            gs_del_flag = False
            if e_count!=0 or p_count!=0 or d_count!=0: gs_del_flag = True  #no excitation/no f state/no decay/

            if  gs_del_flag :
                self.pos_to_del_gs.append(i)
        self.pos_to_del_gs = list(dict.fromkeys(self.pos_to_del_gs)) #remove duplicates
        
        self.gs_dim = self.gs_e1_dec_dim - len(self.pos_to_del_gs)

        self.pos_gs = [i for i in [*range(self.gs_e1_dec_dim )] if i not in self.pos_to_del_gs]  #all positions that contain gs in gs_e1_dec

                
        self.gs_states = np.delete(self.gs_e1_dec_states, self.pos_to_del_gs )
        self.gs_excitations = np.delete(self.gs_e1_dec_excitations, self.pos_to_del_gs )


        # e1_hamiltonian states and positions in gs_e1_dec subspace
        
        self.pos_to_del_e1 = []
        for (i , excitation ) in enumerate(self.gs_e1_dec_excitations) : 
            letter_count = Counter(excitation)
            q_count = letter_count['q'] #g states
            e_count = letter_count['e'] #e/E states
            d_count = letter_count['d'] #o states

            e1_del_flag = False
            if e_count!=1 or q_count!=0 or d_count!=0: e1_del_flag = True  #1 exc /no g state/no decay  
 

            if e1_del_flag :
                self.pos_to_del_e1.append(i)
        self.pos_to_del_e1 = list(dict.fromkeys(self.pos_to_del_e1)) #remove duplicates
        
        self.e1_dim = self.gs_e1_dec_dim - len(self.pos_to_del_e1)  

        self.pos_e1 = [i for i in [*range(self.gs_e1_dec_dim )] if i not in self.pos_to_del_e1]  #all positions that contain e1 in gs_e1_dec

        self.e1_states = np.delete(self.gs_e1_dec_states, self.pos_to_del_e1 )
        self.e1_excitations = np.delete(self.gs_e1_dec_excitations, self.pos_to_del_e1 )
      

        #dec states in the subspace
        self.pos_to_del_dec = []
        for (i , excitation ) in enumerate(self.gs_e1_dec_excitations) : 
            letter_count = Counter(excitation)
            q_count = letter_count['q'] #g states
            e_count = letter_count['e'] #e/E states
            d_count = letter_count['d'] #o states

            dec_del_flag = False
            if e_count!=0 or q_count!=0 or d_count>1: dec_del_flag = True  #0 exc /no g state/ 1 or 0 decay  

            if  dec_del_flag:
                self.pos_to_del_dec.append(i)
        self.pos_to_del_dec = list(dict.fromkeys(self.pos_to_del_dec)) #remove duplicates
        
        self.dec_dim = self.gs_e1_dec_dim - len(self.pos_to_del_dec)  

        self.pos_dec = [i for i in [*range(self.gs_e1_dec_dim )] if i not in self.pos_to_del_dec]  #all positions that contain dec in gs_e1_dec

        self.dec_states = np.delete(self.gs_e1_dec_states, self.pos_to_del_dec )
        self.dec_excitations = np.delete(self.gs_e1_dec_excitations, self.pos_to_del_dec )


        self.gs_e1_dec_matrix_space = sg.MatrixSpace( sg.SR ,self.gs_e1_dec_dim,self.gs_e1_dec_dim ,sparse=False ) 



    def obtain_energy_info(self):
        '''
        Fetches Hamiltonians and Lindblau operators from components.
        '''
        self.H_list = []
        self.H_coeffs = []
        self.gs_e1_dec_int =[]
        
        self.Lindblau_list = []
        self.L_coeffs = []
        
        for elem in self.elements:
            for sub_elem in elem.sub_elements:
                
                h = sub_elem.hamiltonian()             
                for (i,h_el) in enumerate(h):
                    self.H_list.append(h[i])
                    self.H_coeffs.append(sub_elem.H_coeffs[i])
                    self.gs_e1_dec_int.append(sub_elem.gs_e1_interaction[i])


                lind = sub_elem.lindblau() 
                for (i,l_el) in enumerate(lind):
                    self.Lindblau_list.append(lind[i])
                    self.L_coeffs.append(sub_elem.L_coeffs[i])



    def construct_gs_hamiltonian(self):
        '''
        Constructs ground state Hamiltonian in gs_e1_dec subspace, corresponding state-vectors and corresponding excitation.

        Note that the gs_hamiltonian will be a numpy array and not a qt objeect.
        '''
        self.gs_hamiltonian = np.zeros((self.gs_e1_dec_dim,self.gs_e1_dec_dim) , dtype = 'complex128')

        self.gs_hamiltonian = sg.matrix(self.gs_hamiltonian )

        for (coeff , h) in zip(self.H_coeffs,self.H_list):
            h_reduced = delete_from_csr( h.data, row_indices=self.pos_to_del_gs_e1_dec, col_indices=self.pos_to_del_gs_e1_dec).toarray() 
            h_reduced[self.pos_e1, :]  = 0
            h_reduced[: , self.pos_e1] = 0
            h_reduced[self.pos_dec, :]  = 0
            h_reduced[: , self.pos_dec] = 0
            self.gs_hamiltonian = self.gs_hamiltonian + coeff * sg.matrix(h_reduced)
        
        ones_w_0diag = np.ones((self.gs_e1_dec_dim,self.gs_e1_dec_dim))
        np.fill_diagonal(ones_w_0diag , 0)
        ones_w_0diag = sg.matrix(ones_w_0diag ) + sg.var('x')*sg.matrix(np.zeros((self.gs_e1_dec_dim,self.gs_e1_dec_dim)))

                 


        self.gs_hamiltonian =  self.gs_hamiltonian   + elementwise(sg.operator.mul, self.gs_hamiltonian , ones_w_0diag).conjugate_transpose()

        self.gs_hamiltonian = self.gs_e1_dec_matrix_space(self.gs_hamiltonian)



    def construct_e1_hamiltonian(self):
        '''
        Constructs the first excited state Hamiltonian in gs_e1_dec subspace, corresponding state-vectors and corresponding excitation.

        Note that the e1_hamiltonian will be a numpy array and not a qt objeect.
        '''
        
        self.e1_hamiltonian = sg.matrix ( np.zeros((self.gs_e1_dec_dim,self.gs_e1_dec_dim), dtype = 'complex128') )
        for (coeff , h) in zip(self.H_coeffs,self.H_list):
            h_reduced = delete_from_csr( h.data, row_indices=self.pos_to_del_gs_e1_dec, col_indices=self.pos_to_del_gs_e1_dec).toarray()
            h_reduced[self.pos_gs, :]  = 0
            h_reduced[: , self.pos_gs] = 0
            h_reduced[self.pos_dec, :]  = 0
            h_reduced[: , self.pos_dec] = 0       
            self.e1_hamiltonian += coeff * sg.matrix( h_reduced  )     

        ones_w_0diag = np.ones((self.gs_e1_dec_dim,self.gs_e1_dec_dim))
        np.fill_diagonal(ones_w_0diag , 0)
        ones_w_0diag = sg.matrix(ones_w_0diag ) + sg.var('x')*sg.matrix(np.zeros((self.gs_e1_dec_dim,self.gs_e1_dec_dim)))

        self.e1_hamiltonian = self.e1_hamiltonian   + elementwise(sg.operator.mul, self.e1_hamiltonian , ones_w_0diag).conjugate_transpose()

        self.e1_hamiltonian = self.gs_e1_dec_matrix_space(self.e1_hamiltonian)



    def construct_V(self):
        '''
        Constructs  V+ and V- in gs_e1_dec subspace, corresponding state-vectors and corresponding excitation.

        Note that the e1_hamiltonian will be a numpy array and not a qt objeect.
        '''

        self.V_plus = sg.matrix( np.zeros((self.gs_e1_dec_dim,self.gs_e1_dec_dim) , dtype = 'complex128')  ) * sg.var('x')

        for (coeff , h , gs_e1_interaction) in zip(self.H_coeffs,self.H_list , self.gs_e1_dec_int):
            if gs_e1_interaction:
                h_reduced = delete_from_csr( h.data, row_indices=self.pos_to_del_gs_e1_dec, col_indices=self.pos_to_del_gs_e1_dec).toarray()
                self.V_plus += coeff * sg.matrix(h_reduced)

        
        self.V_minus = self.V_plus.conjugate_transpose()

        self.V_plus = self.gs_e1_dec_matrix_space(self.V_plus)
        self.V_minus = self.gs_e1_dec_matrix_space(self.V_minus)
        

    def construct_nj_hamiltonian(self):
        '''
        Constructs the nj Hamiltonian.
        '''

        self.L_sum =  sg.copy(self.V_plus.parent().zero())

        for (coeff , lindblau) in zip(self.L_coeffs ,self.Lindblau_list):
            l_reduced = delete_from_csr( lindblau.data, row_indices=self.pos_to_del_gs_e1_dec, col_indices=self.pos_to_del_gs_e1_dec).toarray()      
            L = coeff * self.gs_e1_dec_matrix_space(l_reduced)
            self.L_sum +=  L.conjugate_transpose() * L 
        

        self.nj_hamiltonian = self.e1_hamiltonian - sg.I /2 * self.L_sum



    def construct_nj_hamiltonian_inverse(self):
        '''
        Constructs nj_hamiltonian_inverse.
        
        Finds zero (row and columns) that make the array non singular. 
        Add 1 on the diagonal, then invert  and then set the elements back to zero.
        '''
        self.nj_hamiltonian_inv = sg.copy(self.nj_hamiltonian)
        zero_pos = []
        for i in range(self.gs_e1_dec_dim):         
            if self.nj_hamiltonian_inv[i,:].is_zero()  and self.nj_hamiltonian_inv[i,:].is_zero() :
                zero_pos.append(i)
                self.nj_hamiltonian_inv[i,i] = 1

        self.nj_hamiltonian_inv = self.nj_hamiltonian_inv.inverse()

        #revert it back to its original form
        for i in zero_pos:         
            self.nj_hamiltonian_inv[i,i] = 0


    def construct_eff_hamiltonian_lindblaus(self):
        '''
        Consrtucts effective hamiltonian.
        '''
        self.eff_hamiltonian = sg.copy(self.gs_hamiltonian)
        self.eff_hamiltonian += -1/2*self.V_minus * ( self.nj_hamiltonian_inv +self.nj_hamiltonian_inv.conjugate_transpose() ) * self.V_plus

        #effective operator on gs
        self.eff_hamiltonian_gs = sg.copy(self.eff_hamiltonian )
        self.eff_hamiltonian_gs = self.eff_hamiltonian_gs[self.pos_gs,self.pos_gs]
        
        self.eff_lindblau_list = []
        for (coeff , lindblau) in zip(self.L_coeffs ,self.Lindblau_list):
            l_reduced = delete_from_csr( lindblau.data, row_indices=self.pos_to_del_gs_e1_dec, col_indices=self.pos_to_del_gs_e1_dec).toarray()      
            L_eff = coeff * sg.matrix( l_reduced  ) * self.nj_hamiltonian_inv * self.V_plus
            self.eff_lindblau_list.append( L_eff )

    
    def solve_master_equation(self):
        self.rho_matrix = sg.copy( self.eff_hamiltonian_gs.parent().zero())

        for i in range(self.gs_dim):
            for j in range(self.gs_dim):
                self.rho_matrix[i,j] = sg.var( f'rho_{i}{j}' , domain = 'real' , latex_name = f'\\rho_{{ {i}{j} }}'  )

        









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


        