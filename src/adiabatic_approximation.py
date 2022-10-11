#
# File containing system class that obtains effective hamiltonians and effective Lindblad operators.
#

import numpy 
import sympy 
from collections import Counter
import time

from .abstraction import QOpsSystem
from . import q_ops_utilities



class EffectiveOperatorFormalism():
    '''
    
    '''
 
    def __init__(self, q_ops_system : QOpsSystem ,  cores = None):
        self.cores = cores
        self.q_ops_system = q_ops_system
        # Creation of the system comprised of the elements that make up the system.
        # Elements being o, x, O , -
        self.components = self.q_ops_system.components
        self.dimensions = self.q_ops_system.dimensions
        self.dimension =  numpy.prod(self.q_ops_system.dimensions)

        t_start = time.time()
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
        for component in self.components.values():
            excitations_list.append([])
            states_list.append([])
            for energy_level in component.energy_levels.values():
                excitations_list[-1].append(energy_level.excitation_status)
                states_list[-1].append(energy_level.name)

        for (i,dim) in enumerate(self.dimensions):
            excitations = excitations_list[i]
            states = states_list[i]
            above_dims = numpy.prod(self.dimensions[:i+1]) 
            consecutive_elems = self.dimension // above_dims  
            k = 0
            while k<self.dimension:
                for d in range(dim):
                    for _ in range(consecutive_elems):
                        self.excitations[k] += '|'+ excitations[d]
                        self.states[k] += '|'+ states[d]
                        k = k + 1
        
        for d in range(self.dimension):
            self.excitations[d] += '|'
            self.states[d] += '|'
                

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
            d_count = letter_count['D'] # number of decayed states
            e_count = letter_count['E'] # number of excited states

            if d_count + e_count > 1:
                self.pos_to_del_gs_e1_dec.append(i)
        
        #self.pos_gs_e1_dec = [i for i in [*range(self.dim)] if i not in self.pos_to_del_gs_e1_dec]   #DELETED DUE TO IT BEING TOO SLOW
        self.gs_e1_dec_dim = self.dimension- len(self.pos_to_del_gs_e1_dec)
        self.gs_e1_dec_excitations = numpy.delete(self.excitations , self.pos_to_del_gs_e1_dec)
        self.gs_e1_dec_states = numpy.delete(self.states , self.pos_to_del_gs_e1_dec)


        # gs_hamiltonian states and positions in gs_e1_dec subspace

        self.pos_to_del_gs = []
        for (i , excitation ) in enumerate(self.gs_e1_dec_excitations) :            
            letter_count = Counter(excitation)
            d_count = letter_count['D'] # number of decayed states
            e_count = letter_count['E'] # number of excited states
            
            if  e_count + d_count > 0 :
                self.pos_to_del_gs.append(i)
        
        self.pos_to_del_gs = list(dict.fromkeys(self.pos_to_del_gs)) #remove duplicates
        
        self.gs_dim = self.gs_e1_dec_dim - len(self.pos_to_del_gs)

        self.pos_gs = [i for i in [*range(self.gs_e1_dec_dim )] if i not in self.pos_to_del_gs]  #all positions that contain gs in gs_e1_dec

                
        self.gs_states = numpy.delete(self.gs_e1_dec_states, self.pos_to_del_gs )
        self.gs_excitations = numpy.delete(self.gs_e1_dec_excitations, self.pos_to_del_gs )


        # e1_hamiltonian states and positions in gs_e1_dec subspace
        
        self.pos_to_del_e1 = []
        for (i , excitation ) in enumerate(self.gs_e1_dec_excitations) : 
            letter_count = Counter(excitation)
            e_count = letter_count['E'] # number of excited states

            if e_count != 1 :
                self.pos_to_del_e1.append(i)

        self.pos_to_del_e1 = list(dict.fromkeys(self.pos_to_del_e1)) #remove duplicates
        
        self.e1_dim = self.gs_e1_dec_dim - len(self.pos_to_del_e1)  

        self.pos_e1 = [i for i in [*range(self.gs_e1_dec_dim )] if i not in self.pos_to_del_e1]  #all positions that contain e1 in gs_e1_dec

        self.e1_states = numpy.delete(self.gs_e1_dec_states, self.pos_to_del_e1 )
        self.e1_excitations = numpy.delete(self.gs_e1_dec_excitations, self.pos_to_del_e1 )
      

        #dec states in the subspace
        self.pos_to_del_dec = []
        for (i , excitation ) in enumerate(self.gs_e1_dec_excitations) : 
            letter_count = Counter(excitation)
            d_count = letter_count['D'] # number of decayed states

            if  d_count != 1:
                self.pos_to_del_dec.append(i)
        
        self.pos_to_del_dec = list(dict.fromkeys(self.pos_to_del_dec)) #remove duplicates
        
        self.dec_dim = self.gs_e1_dec_dim - len(self.pos_to_del_dec)  

        self.pos_dec = [i for i in [*range(self.gs_e1_dec_dim )] if i not in self.pos_to_del_dec]  #all positions that contain dec in gs_e1_dec

        self.dec_states = numpy.delete(self.gs_e1_dec_states, self.pos_to_del_dec )
        self.dec_excitations = numpy.delete(self.gs_e1_dec_excitations, self.pos_to_del_dec )





    def obtain_energy_info(self):
        '''
        Fetches Hamiltonians and Lindblad operators from components.
        '''
        self.H_list = []
        self.H_coeffs = []
        self.gs_e1_dec_int =[]
        
        self.Lindblad_list = []
        self.L_coeffs = []
        
        for rabi in self.q_ops_system.hamiltonian['rabis'].values():
            qobj = rabi['Qobj']
            coefficient = rabi['coefficient']
            self.H_list.append(qobj)
            self.H_coeffs.append(coefficient)
            self.gs_e1_dec_int.append(True)
        
        for energy_level in self.q_ops_system.hamiltonian['energy_levels'].values():
            qobj = energy_level['Qobj']
            coefficient = energy_level['coefficient']
            self.H_list.append(qobj)
            self.H_coeffs.append(coefficient)
            self.gs_e1_dec_int.append(False)
        
        for coupling in self.q_ops_system.hamiltonian['couplings'].values():
            qobj = coupling['Qobj']
            coefficient = coupling['coefficient']
            self.H_list.append(qobj)
            self.H_coeffs.append(coefficient)
            self.gs_e1_dec_int.append(False)

        for decay in self.q_ops_system.lindblads.values():
            qobj = decay['Qobj']
            coefficient = decay['coefficient']
            self.Lindblad_list.append(qobj)
            self.L_coeffs.append(coefficient)
        
        self.number_of_lindblads = len(self.Lindblad_list)



    def construct_gs_hamiltonian(self):
        '''
        Constructs ground state Hamiltonian in gs_e1_dec subspace, corresponding state-vectors and corresponding excitation.

        Note that the gs_hamiltonian will be a sage matrix and not a qt object.
        '''
        self.gs_hamiltonian = numpy.zeros((self.gs_e1_dec_dim,self.gs_e1_dec_dim) , dtype = 'complex128')

        self.gs_hamiltonian = sympy.Matrix(self.gs_hamiltonian )

        for (coeff , h) in zip(self.H_coeffs,self.H_list):
            h_reduced = q_ops_utilities.delete_from_csr( h.data, row_indices=self.pos_to_del_gs_e1_dec, col_indices=self.pos_to_del_gs_e1_dec).toarray() 
            h_reduced[self.pos_e1, :]  = 0
            h_reduced[: , self.pos_e1] = 0
            h_reduced[self.pos_dec, :]  = 0
            h_reduced[: , self.pos_dec] = 0
            self.gs_hamiltonian = self.gs_hamiltonian + coeff * sympy.Matrix(h_reduced)
        
        # Because hamiltonians are created without the complex conjugate, we have to add the complex conjugate (if it is not diagonal).
        # The routine below takes care of it.
        self.gs_hamiltonian = q_ops_utilities.make_into_hermitian(self.gs_hamiltonian)




    def construct_e1_hamiltonian(self):
        '''
        Constructs the first excited state Hamiltonian in gs_e1_dec subspace, corresponding state-vectors and corresponding excitation.

        Note that the e1_hamiltonian will be a sage matrix and not a qt object.
        '''
        
        self.e1_hamiltonian = sympy.Matrix ( numpy.zeros((self.gs_e1_dec_dim,self.gs_e1_dec_dim), dtype = 'complex128') )
        for (coeff , h) in zip(self.H_coeffs,self.H_list):
            h_reduced = q_ops_utilities.delete_from_csr( h.data, row_indices=self.pos_to_del_gs_e1_dec, col_indices=self.pos_to_del_gs_e1_dec).toarray()
            h_reduced[self.pos_gs, :]  = 0
            h_reduced[: , self.pos_gs] = 0
            h_reduced[self.pos_dec, :]  = 0
            h_reduced[: , self.pos_dec] = 0       
            self.e1_hamiltonian += coeff * sympy.Matrix( h_reduced  )     

        # Because hamiltonians are created without the complex conjugate, we have to add the complex conjugate (if it is not diagonal).
        # The routine below takes care of it.
        self.e1_hamiltonian = q_ops_utilities.make_into_hermitian(self.e1_hamiltonian)



    def construct_V(self):
        '''
        Constructs  V+ and V- in gs_e1_dec subspace, corresponding state-vectors and corresponding excitation.

        Note that the V_plus and V_minus will be a sage matrix and not qt objects.
        '''

        self.V_plus = sympy.zeros(self.gs_e1_dec_dim,self.gs_e1_dec_dim  ) 

        for (coeff , h , gs_e1_interaction) in zip(self.H_coeffs,self.H_list , self.gs_e1_dec_int):
            if gs_e1_interaction:
                h_reduced = q_ops_utilities.delete_from_csr( h.data, row_indices=self.pos_to_del_gs_e1_dec, col_indices=self.pos_to_del_gs_e1_dec).toarray()
                self.V_plus += coeff * sympy.Matrix(h_reduced)

        
        self.V_minus = self.V_plus.H
    
    

    def construct_nj_hamiltonian(self):
        '''
        Constructs the no-jump Hamiltonian from the excited hamiltonian and the lindblad operators.
        '''

        self.L_sum =  sympy.zeros( self.gs_e1_dec_dim,self.gs_e1_dec_dim  ) 

        for (coeff , lindblad) in zip(self.L_coeffs ,self.Lindblad_list):
            l_reduced = q_ops_utilities.delete_from_csr( lindblad.data, row_indices=self.pos_to_del_gs_e1_dec, col_indices=self.pos_to_del_gs_e1_dec).toarray()      
            L = coeff * sympy.Matrix(l_reduced)
            self.L_sum +=  L.H * L 
        

        self.nj_hamiltonian =  self.e1_hamiltonian - sympy.I /2 * self.L_sum

    

    def construct_nj_hamiltonian_inverse(self):
        '''
        Constructs nj_hamiltonian_inverse.
        
        Finds zero (row and columns) that make the array non singular. 
        Inversion of the non-singular sub-array.
        '''
        self.nj_hamiltonian_inv = sympy.zeros(self.nj_hamiltonian.rows,self.nj_hamiltonian.cols)
        non_zero_pos = []
        for i in range(self.gs_e1_dec_dim):
            # check if row is zero
            row = self.nj_hamiltonian[i,:]
            row_is_zero =  row == sympy.zeros(row.shape[0],row.shape[1])   

            # check if column is zero
            col = self.nj_hamiltonian[:,i]
            col_is_zero =  col == sympy.zeros(col.shape[0],col.shape[1])  
            
            if (not row_is_zero) and (not col_is_zero):
                non_zero_pos.append(i)


        non_singular_sub_array = self.nj_hamiltonian[non_zero_pos,non_zero_pos]
        # invert
        inverted_sub_array = non_singular_sub_array.LUsolve(sympy.eye(non_singular_sub_array.cols))    
        # simplify using together()
        inverted_sub_array = q_ops_utilities.together_for_sympy_matrices(inverted_sub_array, processes= self.cores)

        for i,pos_i in enumerate(non_zero_pos):
            for j,pos_j in enumerate(non_zero_pos):
                self.nj_hamiltonian_inv[pos_i,pos_j] = inverted_sub_array[i,j]




    def construct_eff_hamiltonian_lindblads(self):
        '''
        Consrtucts effective hamiltonian and eff_lindblad operators.
        '''
        self.eff_hamiltonian = self.gs_hamiltonian.copy()
        self.eff_hamiltonian += -1/2*self.V_minus * ( self.nj_hamiltonian_inv +self.nj_hamiltonian_inv.H ) * self.V_plus
        #self.eff_hamiltonian = q_ops_utilities.posify_array(self.eff_hamiltonian)

        #effective operator on gs
        self.eff_hamiltonian_gs = self.eff_hamiltonian.copy()
        self.eff_hamiltonian_gs = self.eff_hamiltonian_gs[self.pos_gs,self.pos_gs]
        
        self.lindblad_list = []
        self.eff_lindblad_list = []
        for (coeff , lindblad) in zip(self.L_coeffs ,self.Lindblad_list):
            l_reduced = q_ops_utilities.delete_from_csr( lindblad.data, row_indices=self.pos_to_del_gs_e1_dec, col_indices=self.pos_to_del_gs_e1_dec).toarray()

            self.lindblad_list.append(coeff * sympy.Matrix( l_reduced  ))  
            
            L_eff = coeff * sympy.Matrix( l_reduced  ) * self.nj_hamiltonian_inv * self.V_plus
            #L_eff = q_ops_utilities.posify_array(L_eff)
            self.eff_lindblad_list.append( L_eff )

    