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

        self.components = self.q_ops_system.components
        self.dimensions = self.q_ops_system.dimensions
        self.dimension =  numpy.prod(self.q_ops_system.dimensions)

        t_start = time.time()
        print('Constructing states and excitations ...')
        self.construct_states_and_excitations()
        print('Constructing ground and first-excited statespace ...')
        self.construct_gs_e1_dec_subspace()
        self.obtain_energy_info()    
        print('Constructing hamiltonian interactions ...')  
        self.construct_gs_hamiltonian()
        self.construct_e1_hamiltonian()
        self.construct_V()
        print('Constructing no-jump hamiltonian and inverting it ...')        
        self.construct_no_jump_hamiltonian()
        self.invert_no_jump_hamiltonian()
        print('Constructing effective operators ...')   
        self.construct_eff_hamiltonian_lindblads()

        t_end = time.time()
        print(f'\nObtained {self.q_ops_system.name} effective operators in {round(t_end-t_start , 1)} seconds.')



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
        
        self._indices_to_del_gs_e1_dec = []
        for (i , excitation ) in enumerate(self.excitations) :
            letter_count = Counter(excitation)
            d_count = letter_count['D'] # number of decayed states
            e_count = letter_count['E'] # number of excited states

            if d_count + e_count > 1:
                self._indices_to_del_gs_e1_dec.append(i)
        
        #self.pos_gs_e1_dec = [i for i in [*range(self.dim)] if i not in self.pos_to_del_gs_e1_dec]   #DELETED DUE TO IT BEING TOO SLOW
        self.dimension_gs_e1_dec = self.dimension- len(self._indices_to_del_gs_e1_dec)
        self.excitations_gs_e1_dec = numpy.delete(self.excitations , self._indices_to_del_gs_e1_dec)
        self.states_gs_e1_dec = numpy.delete(self.states , self._indices_to_del_gs_e1_dec)


        # gs_hamiltonian states and positions in gs_e1_dec subspace

        self._indices_to_del_gs = []
        for (i , excitation ) in enumerate(self.excitations_gs_e1_dec) :            
            letter_count = Counter(excitation)
            d_count = letter_count['D'] # number of decayed states
            e_count = letter_count['E'] # number of excited states
            
            if  e_count + d_count > 0 :
                self._indices_to_del_gs.append(i)
        
        self._indices_to_del_gs = list(dict.fromkeys(self._indices_to_del_gs)) #remove duplicates
        
        self.dimension_gs = self.dimension_gs_e1_dec - len(self._indices_to_del_gs)

        self.indices_gs = [i for i in [*range(self.dimension_gs_e1_dec )] if i not in self._indices_to_del_gs]  #all positions that contain gs in gs_e1_dec

                
        self.states_gs = numpy.delete(self.states_gs_e1_dec, self._indices_to_del_gs )
        self.excitations_gs = numpy.delete(self.excitations_gs_e1_dec, self._indices_to_del_gs )


        # e1_hamiltonian states and positions in gs_e1_dec subspace
        
        self._indices_to_del_e1 = []
        for (i , excitation ) in enumerate(self.excitations_gs_e1_dec) : 
            letter_count = Counter(excitation)
            e_count = letter_count['E'] # number of excited states

            if e_count != 1 :
                self._indices_to_del_e1.append(i)

        self._indices_to_del_e1 = list(dict.fromkeys(self._indices_to_del_e1)) #remove duplicates
        
        self.dimension_e1 = self.dimension_gs_e1_dec - len(self._indices_to_del_e1)  

        self.indices_e1 = [i for i in [*range(self.dimension_gs_e1_dec )] if i not in self._indices_to_del_e1]  #all positions that contain e1 in gs_e1_dec

        self.states_e1 = numpy.delete(self.states_gs_e1_dec, self._indices_to_del_e1 )
        self.excitations_e1 = numpy.delete(self.excitations_gs_e1_dec, self._indices_to_del_e1 )
      

        #dec states in the subspace
        self._indices_to_del_dec = []
        for (i , excitation ) in enumerate(self.excitations_gs_e1_dec) : 
            letter_count = Counter(excitation)
            d_count = letter_count['D'] # number of decayed states

            if  d_count != 1:
                self._indices_to_del_dec.append(i)
        
        self._indices_to_del_dec = list(dict.fromkeys(self._indices_to_del_dec)) #remove duplicates
        
        self.dimension_dec = self.dimension_gs_e1_dec - len(self._indices_to_del_dec)  

        self.indices_dec = [i for i in [*range(self.dimension_gs_e1_dec )] if i not in self._indices_to_del_dec]  #all positions that contain dec in gs_e1_dec

        self.states_dec = numpy.delete(self.states_gs_e1_dec, self._indices_to_del_dec )
        self.excitations_dec = numpy.delete(self.excitations_gs_e1_dec, self._indices_to_del_dec )





    def obtain_energy_info(self):
        '''
        Fetches Hamiltonians and Lindblad operators from components.
        '''
        self.H_list = []
        self.H_coeffs = []
        self.gs_e1_int =[]
        
        self.Lindblad_list = []
        self.L_coeffs = []
        
        for rabi in self.q_ops_system.hamiltonian['rabis'].values():
            qobj = rabi['Qobj']
            coefficient = rabi['coefficient']
            
            rabi_object = rabi['interaction_object']

            self.H_list.append(qobj)
            self.H_coeffs.append(coefficient)

            self.gs_e1_int.append(True)
        
        for energy_level in self.q_ops_system.hamiltonian['energy_levels'].values():
            qobj = energy_level['Qobj']
            coefficient = energy_level['coefficient']
            self.H_list.append(qobj)
            self.H_coeffs.append(coefficient)
            self.gs_e1_int.append(False)
        
        for coupling in self.q_ops_system.hamiltonian['couplings'].values():
            qobj = coupling['Qobj']
            coefficient = coupling['coefficient']
            self.H_list.append(qobj)
            self.H_coeffs.append(coefficient)
            self.gs_e1_int.append(False)

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
        self.hamiltonian_gs = numpy.zeros((self.dimension_gs_e1_dec,self.dimension_gs_e1_dec) , dtype = 'complex128')

        self.hamiltonian_gs = sympy.SparseMatrix(self.hamiltonian_gs )

        for (coeff , h) in zip(self.H_coeffs,self.H_list):
            h_reduced = q_ops_utilities.delete_from_csr( h.data, row_indices=self._indices_to_del_gs_e1_dec, col_indices=self._indices_to_del_gs_e1_dec).toarray() 
            h_reduced[self.indices_e1, :]  = 0
            h_reduced[: , self.indices_e1] = 0
            h_reduced[self.indices_dec, :]  = 0
            h_reduced[: , self.indices_dec] = 0
            self.hamiltonian_gs = self.hamiltonian_gs + coeff * sympy.SparseMatrix(h_reduced)
        
        # Because hamiltonians are created without the complex conjugate, we have to add the complex conjugate (if it is not diagonal).
        # The routine below takes care of it.
        self.hamiltonian_gs = q_ops_utilities.make_into_hermitian(self.hamiltonian_gs)




    def construct_e1_hamiltonian(self):
        '''
        Constructs the first excited state Hamiltonian in gs_e1_dec subspace, corresponding state-vectors and corresponding excitation.

        Note that the e1_hamiltonian will be a sage matrix and not a qt object.
        '''
        
        self.hamiltonian_e1 = sympy.SparseMatrix ( numpy.zeros((self.dimension_gs_e1_dec,self.dimension_gs_e1_dec), dtype = 'complex128') )
        for (coeff , h) in zip(self.H_coeffs,self.H_list):
            h_reduced = q_ops_utilities.delete_from_csr( h.data, row_indices=self._indices_to_del_gs_e1_dec, col_indices=self._indices_to_del_gs_e1_dec).toarray()
            h_reduced[self.indices_gs, :]  = 0
            h_reduced[: , self.indices_gs] = 0
            h_reduced[self.indices_dec, :]  = 0
            h_reduced[: , self.indices_dec] = 0       
            self.hamiltonian_e1 += coeff * sympy.SparseMatrix( h_reduced  )     

        # Because hamiltonians are created without the complex conjugate, we have to add the complex conjugate (if it is not diagonal).
        # The routine below takes care of it.
        self.hamiltonian_e1 = q_ops_utilities.make_into_hermitian(self.hamiltonian_e1)



    def construct_V(self):
        '''
        Constructs  V+ and V- in gs_e1_dec subspace, corresponding state-vectors and corresponding excitation.

        Note that the V_plus and V_minus will be a sage matrix and not qt objects.
        '''

        self.V_plus = sympy.zeros(self.dimension_gs_e1_dec,self.dimension_gs_e1_dec  ) 

        for (coeff , h , gs_e1_interaction) in zip(self.H_coeffs,self.H_list , self.gs_e1_int):
            if gs_e1_interaction:
                h_reduced = q_ops_utilities.delete_from_csr( h.data, row_indices=self._indices_to_del_gs_e1_dec, col_indices=self._indices_to_del_gs_e1_dec).toarray()
                self.V_plus += coeff * sympy.SparseMatrix(h_reduced)

        
        self.V_minus = self.V_plus.H
    
    

    def construct_no_jump_hamiltonian(self):
        '''
        Constructs the no-jump Hamiltonian from the excited hamiltonian and the lindblad operators.
        '''

        L_sum =  sympy.zeros( self.dimension_gs_e1_dec,self.dimension_gs_e1_dec  ) 

        for (coeff , lindblad) in zip(self.L_coeffs ,self.Lindblad_list):
            l_reduced = q_ops_utilities.delete_from_csr( lindblad.data, row_indices=self._indices_to_del_gs_e1_dec, col_indices=self._indices_to_del_gs_e1_dec).toarray()      
            L = coeff * sympy.SparseMatrix(l_reduced)
            L_sum +=  L.H * L 
        

        self.hamiltonian_nj =  self.hamiltonian_e1 - sympy.I /2 * L_sum

    

    def invert_no_jump_hamiltonian(self):
        '''
        Constructs nj_hamiltonian_inverse.
        
        Finds zero (row and columns) that make the array non singular. 
        Inversion of the non-singular sub-array.
        '''
        self.nj_hamiltonian_inv = sympy.zeros(self.hamiltonian_nj.rows,self.hamiltonian_nj.cols)
        
        non_zero_indices_nj_ham = []
        for i in range(self.dimension_gs_e1_dec):
            # check if row is zero
            row = self.hamiltonian_nj[i,:]
            row_is_zero =  row == sympy.zeros(row.shape[0],row.shape[1])   

            # check if column is zero
            col = self.hamiltonian_nj[:,i]
            col_is_zero =  col == sympy.zeros(col.shape[0],col.shape[1])  
            
            if (not row_is_zero) and (not col_is_zero):
                non_zero_indices_nj_ham.append(i)


        non_singular_sub_array = self.hamiltonian_nj[non_zero_indices_nj_ham,non_zero_indices_nj_ham]
        # invert
        inverted_sub_array = non_singular_sub_array.LUsolve(sympy.eye(non_singular_sub_array.cols))    
        # simplify using together()
        inverted_sub_array = q_ops_utilities.together_for_sympy_matrices(inverted_sub_array, processes= self.cores)

        for i,pos_i in enumerate(non_zero_indices_nj_ham):
            for j,pos_j in enumerate(non_zero_indices_nj_ham):
                self.nj_hamiltonian_inv[pos_i,pos_j] = inverted_sub_array[i,j]




    def construct_eff_hamiltonian_lindblads(self):
        '''
        Consrtucts effective hamiltonian and eff_lindblad operators.
        '''
        self.effective_hamiltonian = self.hamiltonian_gs.copy()
        
        self.effective_hamiltonian += -1/2*self.V_minus * ( self.nj_hamiltonian_inv +self.nj_hamiltonian_inv.H ) * self.V_plus

        #effective operator on gs
        self.effective_hamiltonian_gs = self.effective_hamiltonian.copy()
        self.effective_hamiltonian_gs = self.effective_hamiltonian_gs[self.indices_gs,self.indices_gs]
        
        self.lindblad_list = []
        self.effective_lindblad_list = []
        for (coeff , lindblad) in zip(self.L_coeffs ,self.Lindblad_list):
            l_reduced = q_ops_utilities.delete_from_csr( lindblad.data, row_indices=self._indices_to_del_gs_e1_dec, col_indices=self._indices_to_del_gs_e1_dec).toarray()

            self.lindblad_list.append(coeff * sympy.SparseMatrix( l_reduced  ))  
            
            L_eff = coeff * sympy.SparseMatrix( l_reduced  ) * self.nj_hamiltonian_inv * self.V_plus
            #L_eff = q_ops_utilities.posify_array(L_eff)
            self.effective_lindblad_list.append( L_eff )

    