#
# File containing system class that obtains effective hamiltonians and effective Lindblad operators.
#


import numpy as np
import sympy as sp
from collections import Counter
import time

from .abstraction import QOpsSystem
from . import system_functions



class EffectiveOperatorFormalism():
    '''
    Class defining a system of elements.

    
    Corresponding state-vectors and corresponding excitations will be initialized.

    Extras:

    ManyVariables = False : bool
        Different cavities have different detunings 
    TwoPhotonResonance = True: bool
        Cavities are on two photon resonance

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
    
    
    '''
 
    def __init__(self, q_ops_system : QOpsSystem ,  cores = None):
        self.cores = cores
        self.q_ops_system = q_ops_system
        # Creation of the system comprised of the elements that make up the system.
        # Elements being o, x, O , -
        self.components = self.q_ops_system.components
        self.dimensions = self.q_ops_system.dimensions
        self.dimension =  np.prod(self.q_ops_system.dimensions)

        t_start = time.time()
        self.update_subelements()
        print('Constructing states and excitations ...')
        self.construct_states_and_excitations()
        print('Constructing ground and first-excited statespace ...')
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
        print('Inverting NJ_hamiltonian ...') 
        self.construct_nj_hamiltonian_inverse()
        print('Constructing eff_hamiltonian and effective lindblad operators ...')   
        self.construct_eff_hamiltonian_lindblads()

        
        t_end = time.time()
        print(f'\nQOpsSystem {self.q_ops_system.name} effective operators in {round(t_end-t_start , 1)} seconds.')



    def construct_states_and_excitations(self):
        '''
        Creates the 'str' vectors self.excitations  and  self.states.
        Each entry will contain a string representing the excitation/state of each subelement in the system.

        For example, in o-x it will be a string of length 5 (2 cavities, 2 atoms and 1 fiber.)
        '''
        self.excitations = ['']*self.dimension
        self.states = ['']*self.dimension

        #This code does "tensor product" for characters
        excitations_list = []
        states_list = []
        for elem in self.elements:
            for sub_elem in elem.sub_elements:
                excitations_list.append(sub_elem.excitations)
                states_list.append(sub_elem.states)

        for (i,dim) in enumerate(self.dimensions):
            excitations = excitations_list[i]
            states = states_list[i]
            above_dims = np.prod(self.dimensions[:i+1]) 
            consecutive_elems = self.dimension // above_dims  
            k = 0
            while k< self.dim:
                for d in range(dim):
                    for c in range(consecutive_elems):
                        self.excitations[k] += excitations[d] 
                        self.states[k] += states[d] 
                        k = k + 1

                
    def construct_gs_e1_dec_subspace(self):
        '''
        Constructs: 

        self.pos_to_del_gs_e1_dec  : positions from full Hamiltionian to delete to obtain the subspace gs_e1_dec
        self.gs_e1_dec_excitations : the excitation for each of the dimensions
        self.gs_e1_dec_states      : the statevector represented by a string for each of the dimensions
        
        WITHIN THE GS_E1_DEC subspace:
            Ground state
            self.pos_gs                : all positions that contain gs in gs_e1_dec
            self.pos_to_del_gs         : positions from gs_e1_dec_Hamiltionian to delete to obtain the subspace gs
            self.gs_excitations        : the excitation for each of the dimensions
            self.gs_dec_states         : the statevector represented by a string for each of the dimensions

            Similarly for e1 and dec...
        '''
        
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
        
        #self.pos_gs_e1_dec = [i for i in [*range(self.dim)] if i not in self.pos_to_del_gs_e1_dec]   #DELETED DUE TO IT BEING TOO SLOW
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





    def obtain_energy_info(self):
        '''
        Fetches Hamiltonians and Lindblad operators from components.
        '''
        self.H_list = []
        self.H_coeffs = []
        self.gs_e1_dec_int =[]
        
        self.Lindblad_list = []
        self.L_coeffs = []
        
        for elem in self.elements:
            for sub_elem in elem.sub_elements:
                
                h = sub_elem.hamiltonian()             
                for (i,h_el) in enumerate(h):
                    self.H_list.append(h[i])
                    self.H_coeffs.append(sub_elem.H_coeffs[i])
                    self.gs_e1_dec_int.append(sub_elem.gs_e1_interaction[i])


                lind = sub_elem.lindblad() 
                for (i,l_el) in enumerate(lind):
                    self.Lindblad_list.append(lind[i])
                    self.L_coeffs.append(sub_elem.L_coeffs[i])
        
        self.number_of_lindblads = len(self.Lindblad_list)



    def construct_gs_hamiltonian(self):
        '''
        Constructs ground state Hamiltonian in gs_e1_dec subspace, corresponding state-vectors and corresponding excitation.

        Note that the gs_hamiltonian will be a sage matrix and not a qt object.
        '''
        self.gs_hamiltonian = np.zeros((self.gs_e1_dec_dim,self.gs_e1_dec_dim) , dtype = 'complex128')

        self.gs_hamiltonian = sp.Matrix(self.gs_hamiltonian )

        for (coeff , h) in zip(self.H_coeffs,self.H_list):
            h_reduced = system_functions.delete_from_csr( h.data, row_indices=self.pos_to_del_gs_e1_dec, col_indices=self.pos_to_del_gs_e1_dec).toarray() 
            h_reduced[self.pos_e1, :]  = 0
            h_reduced[: , self.pos_e1] = 0
            h_reduced[self.pos_dec, :]  = 0
            h_reduced[: , self.pos_dec] = 0
            self.gs_hamiltonian = self.gs_hamiltonian + coeff * sp.Matrix(h_reduced)
        
        # Because hamiltonians are created without the complex conjugate, we have to add the complex conjugate (if it is not diagonal).
        # The routine below takes care of it.
        self.gs_hamiltonian = system_functions.make_into_hermitian(self.gs_hamiltonian)




    def construct_e1_hamiltonian(self):
        '''
        Constructs the first excited state Hamiltonian in gs_e1_dec subspace, corresponding state-vectors and corresponding excitation.

        Note that the e1_hamiltonian will be a sage matrix and not a qt object.
        '''
        
        self.e1_hamiltonian = sp.Matrix ( np.zeros((self.gs_e1_dec_dim,self.gs_e1_dec_dim), dtype = 'complex128') )
        for (coeff , h) in zip(self.H_coeffs,self.H_list):
            h_reduced = system_functions.delete_from_csr( h.data, row_indices=self.pos_to_del_gs_e1_dec, col_indices=self.pos_to_del_gs_e1_dec).toarray()
            h_reduced[self.pos_gs, :]  = 0
            h_reduced[: , self.pos_gs] = 0
            h_reduced[self.pos_dec, :]  = 0
            h_reduced[: , self.pos_dec] = 0       
            self.e1_hamiltonian += coeff * sp.Matrix( h_reduced  )     

        # Because hamiltonians are created without the complex conjugate, we have to add the complex conjugate (if it is not diagonal).
        # The routine below takes care of it.
        self.e1_hamiltonian = system_functions.make_into_hermitian(self.e1_hamiltonian)



    def construct_V(self):
        '''
        Constructs  V+ and V- in gs_e1_dec subspace, corresponding state-vectors and corresponding excitation.

        Note that the V_plus and V_minus will be a sage matrix and not qt objects.
        '''

        self.V_plus = sp.zeros(self.gs_e1_dec_dim,self.gs_e1_dec_dim  ) 

        for (coeff , h , gs_e1_interaction) in zip(self.H_coeffs,self.H_list , self.gs_e1_dec_int):
            if gs_e1_interaction:
                h_reduced = system_functions.delete_from_csr( h.data, row_indices=self.pos_to_del_gs_e1_dec, col_indices=self.pos_to_del_gs_e1_dec).toarray()
                self.V_plus += coeff * sp.Matrix(h_reduced)

        
        self.V_minus = self.V_plus.H
    
    

    def construct_nj_hamiltonian(self):
        '''
        Constructs the no-jump Hamiltonian from the excited hamiltonian and the lindblad operators.
        '''

        self.L_sum =  sp.zeros( self.gs_e1_dec_dim,self.gs_e1_dec_dim  ) 

        for (coeff , lindblad) in zip(self.L_coeffs ,self.Lindblad_list):
            l_reduced = system_functions.delete_from_csr( lindblad.data, row_indices=self.pos_to_del_gs_e1_dec, col_indices=self.pos_to_del_gs_e1_dec).toarray()      
            L = coeff * sp.Matrix(l_reduced)
            self.L_sum +=  L.H * L 
        

        self.nj_hamiltonian =  self.e1_hamiltonian - sp.I /2 * self.L_sum

    

    def construct_nj_hamiltonian_inverse(self):
        '''
        Constructs nj_hamiltonian_inverse.
        
        Finds zero (row and columns) that make the array non singular. 
        Inversion of the non-singular sub-array.
        '''
        self.nj_hamiltonian_inv = sp.zeros(self.nj_hamiltonian.rows,self.nj_hamiltonian.cols)
        non_zero_pos = []
        for i in range(self.gs_e1_dec_dim):
            # check if row is zero
            row = self.nj_hamiltonian[i,:]
            row_is_zero =  row == sp.zeros(row.shape[0],row.shape[1])   

            # check if column is zero
            col = self.nj_hamiltonian[:,i]
            col_is_zero =  col == sp.zeros(col.shape[0],col.shape[1])  
            
            if (not row_is_zero) and (not col_is_zero):
                non_zero_pos.append(i)


        non_singular_sub_array = self.nj_hamiltonian[non_zero_pos,non_zero_pos]
        # invert
        inverted_sub_array = non_singular_sub_array.LUsolve(sp.eye(non_singular_sub_array.cols))    
        # simplify using together()
        inverted_sub_array = system_functions.together_for_sympy_matrices(inverted_sub_array, processes= self.cores)

        for i,pos_i in enumerate(non_zero_pos):
            for j,pos_j in enumerate(non_zero_pos):
                self.nj_hamiltonian_inv[pos_i,pos_j] = inverted_sub_array[i,j]




    def construct_eff_hamiltonian_lindblads(self):
        '''
        Consrtucts effective hamiltonian and eff_lindblad operators.
        '''
        self.eff_hamiltonian = self.gs_hamiltonian.copy()
        self.eff_hamiltonian += -1/2*self.V_minus * ( self.nj_hamiltonian_inv +self.nj_hamiltonian_inv.H ) * self.V_plus
        self.eff_hamiltonian = system_functions.posify_array(self.eff_hamiltonian)

        #effective operator on gs
        self.eff_hamiltonian_gs = self.eff_hamiltonian.copy()
        self.eff_hamiltonian_gs = self.eff_hamiltonian_gs[self.pos_gs,self.pos_gs]
        
        self.lindblad_list = []
        self.eff_lindblad_list = []
        for (coeff , lindblad) in zip(self.L_coeffs ,self.Lindblad_list):
            l_reduced = system_functions.delete_from_csr( lindblad.data, row_indices=self.pos_to_del_gs_e1_dec, col_indices=self.pos_to_del_gs_e1_dec).toarray()

            self.lindblad_list.append(coeff * sp.Matrix( l_reduced  ))  
            
            L_eff = coeff * sp.Matrix( l_reduced  ) * self.nj_hamiltonian_inv * self.V_plus
            L_eff = system_functions.posify_array(L_eff)
            self.eff_lindblad_list.append( L_eff )

    