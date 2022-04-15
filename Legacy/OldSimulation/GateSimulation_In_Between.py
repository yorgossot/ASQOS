#
# File containing all necessary classes to simulate the gate through jupyter notebooks.
#

from system import *
from qutip.qip.operations import rz
from scipy.optimize import minimize
import copy

###############################################  Functions  ##########################################################

maximum_cost = 1

def gate_performance_cost_function(fidelity,p_success,gate_time, ):
    '''
    Cost function to be minimized in order to achieve maximum performance.
    Designed to handle arrays as well.
    The function should be smaller than 1.
    '''

    def performance_function(fidelity,p_success,gate_time):
        '''
        Here the actual cost function is defined.
        '''
        if performance_makes_sense(fidelity, p_success, gate_time):
            '''fidelity_cap = 0.95
            avg_time_to_success = max(gate_time/p_success , 1e-10)
            if fidelity >= fidelity_cap:
                cost = maximum_cost - fidelity_cap - 1/avg_time_to_success
            else:
                cost = maximum_cost - fidelity
            '''
            avg_time_to_success = -1/max(gate_time/p_success , 1e-10)
            cost = avg_time_to_success
        else:
            # if the performance parameters dont make sense, give the maximum cost 
            cost = maximum_cost
        return cost
    
    # handling of arrays
    if type(fidelity) != np.ndarray:
        return performance_function(fidelity,p_success,gate_time)
    else:
        cost_array = np.zeros_like(fidelity)
        for idx, _ in np.ndenumerate(fidelity):
            cost_array[idx] = performance_function(fidelity[idx] ,p_success[idx] ,gate_time[idx] )
        return cost_array


def performance_makes_sense(fidelity,p_success,gate_time):
    '''
    Checks if performance parameters make sense.
    '''
    performance_array = np.array([fidelity,p_success,gate_time])
    if np.any(np.zeros(3) >= performance_array):
        return False
    if np.any(np.full_like(performance_array,np.inf) == performance_array):
        return False
    if fidelity > 1.001 or p_success > 1.001:
        return False
    return True



def concurrence_from_ket(ket):
    '''
    Calculates concurrence for 4-ket of numpy.
    '''
    ket = np.array(ket,dtype=complex)
    qobj = qt.Qobj(ket)
    qobj.dims =  [[2, 2], [1, 1]]
    return qt.concurrence(qobj)


def dictionary_substitution(symoblic_expression, parameter_dictionary):   
    '''
    Takes a symbolic expression and substitutes the variables from the parameter dictionary, returning the result.
    '''

    q = sg.copy(symoblic_expression)
    for var in parameter_dictionary.keys():
        q = eval(f"q.subs({var}= parameter_dictionary['{var}'] )") 
    return q

############################################### Classes ##########################################################


class Simulation():

    def __init__(self,setup_char):
        self.setup_char = setup_char
        self.setup = system(setup_char,MMA=True,ManyVariables=False,TwoPhotonResonance= False)
        self.variables_declaration()
        self.create_parameter_dict()
        print('\nPreparing Superoperator sub-class')
        self.Superoperator = Superoperator(self)
        print('Preparing Analytical sub-class')
        self.Analytical = Analytical(self)
        print('\nDone!')
        
        
    def create_parameter_dict(self):
        '''
        Creates a global parameter setting which shall be used for the simulations. 
        The free parameters shall be afterwards the cooperativities and the detunings.
        '''  
        self.parameters = dict()
        # Variables that will be substituted
        self.variables = ['v','g','g_f','gamma','gamma_g','gamma_f','phi','Omega','kappa_c','kappa_b','g0','gamma0']

        # Values of those parameters
        gamma_val = 1. #*0
        gamma0_val = gamma_val #*0
        
        # Used for analytical results
        gamma_g_val = 0 # DO NOT CHANGE
        gamma_f_val = 1 * gamma_val#*0
        
        # Used for realistic results
        gamma_g_real = 0 * gamma_val#*0
        gamma_f_real = gamma_val - gamma_g_real#*0

        kappa_b_val = 10000* gamma_val#*0
        kappa_c_val = 10000* gamma_val#*0
        
        C = sg.var('C')
        g_val = sg.sqrt( C * kappa_c_val * gamma_val)  #*0 + sg.sqrt( C)
        
        Omega_val =  sg.sqrt(C) * gamma_val * 0.25 / 2 #*0 + sg.sqrt( C)* 0.25 / 2

        phi_val = 0

        g_f_val = g_val
        g0_val  = g_val
        
        c = sg.var('c')
        v_val   = sg.sqrt( c * kappa_c_val * kappa_b_val) #*0 +   sg.sqrt( c)

        for var in self.variables:
            exec(f"self.parameters['{var}'] = {var}_val ")

        self.realistic_parameters = copy.deepcopy(self.parameters)
        self.realistic_parameters['gamma_g'] = gamma_g_real
        self.realistic_parameters['gamma_f'] = gamma_f_real


    def variables_declaration(self):
        sg.var('DEg',domain='positive',  latex_name =r'\Delta_{E\gamma}')
        sg.var('Deg',domain='positive',  latex_name =r'\Delta_{e\gamma}')
        sg.var('c',domain='positive',  latex_name =r'c')
        sg.var('C',domain='positive',  latex_name =r'C')
        sg.var('gamma','DE','De','g','g_f','Omega','v','gamma_f','gamma_g','gamma0','De0','phi','g0','gamma0','dc','db',domain='positive')
        sg.var('r0',domain='real',latex_name =r'r_0')
        sg.var('R_f',domain='real')#ratio  (g_f/g)^2
        sg.var('R0',domain='positive',  latex_name =r'R_0')
        sg.var('R_v',domain='real',  latex_name =r'R_{\nu}') #ratio (v/g)^2
        sg.var('r_g',domain='real',latex_name =r'r_g')
        sg.var('r_b',domain='real')
        sg.var('r_f',domain='real',latex_name =r'r_f')
        sg.var('kappa_c','kappa_b',domain='positive')






class Superoperator():
    '''
    Class designed for superoperator simulations. 
    It is valid no matter the simulation setting as long as the lindblad operators are not time dependant.
    '''
        
    def __init__(self, SimulationClass):
        self.setup = SimulationClass.setup
        self.parameters = SimulationClass.parameters
        self.realistic_parameters = SimulationClass.realistic_parameters
        self.variables = SimulationClass.variables

        self.obtain_lindblads()
        self.project_on_gs_dec_subspace()

        # In between gates as superoperators:
        I = qt.identity(2)
        X = qt.qip.operations.x_gate()
        def ten(A,B):
            ten_oper = qt.identity(self.setup.gs_dim +self.setup.dec_dim).data.toarray()
            AB_ten =  qt.tensor(A,B).data.toarray() 
            for ind_i, i in enumerate(self.GsPosInGsDec):
                for ind_j,j in enumerate(self.GsPosInGsDec):
                    ten_oper[i][j] = AB_ten[ind_i][ind_j]
            return qt.to_super(qt.Qobj(ten_oper))
        
        self.gates_in_betw = [ ten(I,I),ten(I,X),ten(X,X),ten(X,I)]


    def obtain_lindblads(self):
        '''
        Retrieve lindblad operators from the system class.
        '''
        self.lind_op_number = len(self.setup.lindblad_list)
        self.eff_lind = []
        self.eff_lind_master_eq = []
        self.eff_lind_coeff = []
        for lind_op in range(self.lind_op_number ):
            self.eff_lind_coeff.append(self.setup.L_coeffs[lind_op])
            L_meq =  self.setup.eff_lindblad_list[lind_op]
            
            self.eff_lind_master_eq.append(L_meq)

    def project_on_gs_dec_subspace(self):
        '''
        This function eliminates the first excited state to facilitate the simulations.
        New indexing is calculated before the projection. 
        '''
        
        GsE1DecPos = np.arange(self.setup.gs_e1_dec_dim)

        GsDecPos = np.delete(GsE1DecPos ,self.setup.pos_e1)

        self.GsPosInGsDec = []
        for p in self.setup.pos_gs:
            where = np.where(GsDecPos == p)[0][0]
            self.GsPosInGsDec.append(where)
        self.GsPosInGsDec = np.array(self.GsPosInGsDec)
                       
        self.DecPosInGsDec = []
        for p in self.setup.pos_dec:
            where = np.where(GsDecPos == p)[0][0]
            self.DecPosInGsDec.append(where)
        self.DecPosInGsDec = np.array(self.DecPosInGsDec)                    
        
        #project the hamiltonian and the lindblads
        self.eff_hamiltonian = sg.Matrix(sg.SR,  self.setup.eff_hamiltonian[ [k for k in GsDecPos] ,  [k for k in GsDecPos]] )

        for lind_op in range(self.lind_op_number ):  
            self.eff_lind_master_eq[lind_op] = sg.Matrix(sg.SR,  self.eff_lind_master_eq[lind_op][ [k for k in GsDecPos] ,  [k for k in GsDecPos]] )

    
    def basic_substitution(self, realistic):
        '''
        Substitutes into the Hamiltonian and the Lindblad operators the self.parameters dictionary values.
        '''      
        def loc_substitute(q , realistic):
            if realistic:
                q = dictionary_substitution(q,self.realistic_parameters)
            else:
                q = dictionary_substitution(q,self.parameters)
            return q
            
        self.eff_hamiltonian_C =  loc_substitute(self.eff_hamiltonian, realistic) 

        self.eff_lind_master_eq_C = []
        for lind_op in range(self.lind_op_number):
            self.eff_lind_master_eq_C.append(  loc_substitute(self.eff_lind_master_eq[lind_op] , realistic )     )


    def Simulate(self, super_cooperativities_dict , super_tg_rots_dict, super_detunings_dict_p , super_detunings_dict_m , realistic):
        '''
        Run simulation for a specified set of values.

        super_dict = dict['C','c','De','De0','DE','dc','db','tgr','r1r','r2r']


        realistic : boolean
        '''
        
        self.basic_substitution(realistic)
        
        super_cooperativities = ['C','c']
        super_detunings = ['De','De0','DE','dc','db']
        super_tg_rots = ['tgr','r1r','r2r']
        
        super_variables_pm = super_cooperativities + super_detunings 
        
        self.super_dict_p =  {**super_cooperativities_dict, **super_detunings_dict_p}

        self.super_dict_m =  {**super_cooperativities_dict, **super_detunings_dict_m}

        eff_hamiltonian_num = dictionary_substitution(self.eff_hamiltonian_C , self.super_dict_p)  

        eff_hamiltonian_num = eff_hamiltonian_num.apply_map(sg.real).numpy().astype(float)


        gs_pos = self.GsPosInGsDec
        H = [eff_hamiltonian_num[gs_pos[0],gs_pos[0]], eff_hamiltonian_num[gs_pos[1],gs_pos[1]]\
            ,eff_hamiltonian_num[gs_pos[2],gs_pos[2]], eff_hamiltonian_num[gs_pos[3],gs_pos[3]]]

        H_eff =[H[0] + H[1]+H[2] -H[3],
                H[0] + H[3],
                H[0] + H[3],
                H[3] + H[1]+H[2] -H[0]]

        H_obj = qt.Qobj(eff_hamiltonian_num)
        self.H_obj = H_obj
        #Initialize superoperator with hamiltonian
        L = -1j*( qt.spre(H_obj) -qt.spost(H_obj))

        L_obj_list = []
        L_np_list = []
        for lind_op in range(len(self.eff_lind_master_eq_C)):
            
            L_nparray = dictionary_substitution( self.eff_lind_master_eq_C[lind_op], self.super_dict_p).numpy().astype(complex)
            L_np_list.append(L_nparray)
            L_obj_list.append(qt.Qobj(L_nparray))
            
            lind = qt.Qobj(L_nparray)
            L += qt.to_super(lind) - 0.5 * (  qt.spre(lind.dag()*lind) + qt.spost(lind.dag()*lind) )
        
        self.L_obj_list = L_obj_list


        # L minus
        eff_hamiltonian_num = dictionary_substitution(self.eff_hamiltonian_C , self.super_dict_m) 
        eff_hamiltonian_num = eff_hamiltonian_num.apply_map(sg.real).numpy().astype(float)
        H_obj = qt.Qobj(eff_hamiltonian_num)
        self.H_obj = H_obj
        #Initialize superoperator with hamiltonian
        L_m = -1j*( qt.spre(H_obj) -qt.spost(H_obj))

        for lind_op in range(len(self.eff_lind_master_eq_C)):
            L_nparray =  dictionary_substitution( self.eff_lind_master_eq_C[lind_op], self.super_dict_p).numpy().astype(complex)
            lind = qt.Qobj(L_nparray)
            L_m += qt.to_super(lind) - 0.5 * (  qt.spre(lind.dag()*lind) + qt.spost(lind.dag()*lind) )

        

        init_state = np.zeros(self.setup.gs_dim +self.setup.dec_dim)
        init_state[self.GsPosInGsDec] = 1/2 # |++> state
        psi0 = qt.Qobj(init_state)
        init_dm = qt.ket2dm(psi0)
        init_vec_dm = qt.operator_to_vector(init_dm)

        psif_gs  =  qt.Qobj(np.array([1,1,1,-1])/2)  #target state

        
        #
        gate_time =  2 * np.abs( np.pi  / ( (H[3]+H[0]-H[1]-H[2]) ) ) * super_tg_rots_dict['tgr'] 

        
        if np.isposinf(gate_time) or np.isneginf(gate_time):
            return  0,1 , 1
        
        #post process rotations
        
        rz = qt.qip.operations.rz

        r1 = - gate_time/4 * (H_eff[0] - H_eff[2]) * super_tg_rots_dict['r1r']           
        R1 = qt.Qobj(qt.tensor( rz(r1 ),qt.identity(2)).full())
        
        
        r2 = -gate_time/4 * (H_eff[0] - H_eff[1]) * super_tg_rots_dict['r2r']  
        R2 = qt.Qobj( qt.tensor(qt.identity(2),rz(r2 )).full())
        
        Lt = (L*gate_time/4).expm()

        L_mt = (L_m*gate_time/4).expm()
        #simulate
        dm_f_vec = init_vec_dm
        
        evol_opers = [L_mt, Lt, Lt,Lt]
        for i in range(4):
            dm_f_vec = self.gates_in_betw[i]*evol_opers[i]* dm_f_vec

        dm_f = qt.vector_to_operator(dm_f_vec)

        
        dec_pos = self.DecPosInGsDec
        
        Psuccess = 1
        for pos in dec_pos:
            Psuccess -= dm_f[pos,pos]
        p_success = np.real(Psuccess)

        dm_f_gs =   1/ Psuccess * dm_f.eliminate_states(dec_pos) #ground state
        
        RotatedFinalState = R1.dag() * R2.dag() * psif_gs
        
        fidelity = qt.fidelity(dm_f_gs,RotatedFinalState)

        dm_f_gs.dims =  [[2, 2], [2, 2]] # Reset the dimensions as 2 qubit tensor so that concurrence function can be called
        concurrence = qt.concurrence(dm_f_gs)


        self.fidelity = fidelity
        self.p_success = p_success
        self.gate_time = gate_time
        self.concurrence = concurrence

        performance = {'fidelity': fidelity , 'p_success': p_success , 'gate_time': gate_time ,'concurrence':concurrence }
        self.performance = {'fidelity': self.fidelity , 'p_success': self.p_success , 'gate_time': self.gate_time,'concurrence':self.concurrence}

        return performance





    def optimize_gate_performance_hardware(self, initial_params , hardware_dict , max_iter = 10):
        '''
        Optimize the gate performance, minimizing some cost function using Superoperator Simulations.
        Initial parameters is the starting point for minimization.

        initial_params = [ De_in, DE_in, tgr_in ]

        hardware_dict = {'C': __, 'c': ___, 'max_split': __} (the final three entries don't matter as they will be used from initial_params)

        max_iter = 10  #maximum number of superoperator simulations
        '''
        


        def cost_function(params, hardware_dict = hardware_dict ):
            '''
            Function that is used for minimization. Essentially it only makes use of global gate performance function.
            '''
            de, dE, tgr , r1r , r2r = params
            super_dict = {'C' : hardware_dict['C'] ,'c':hardware_dict['c'],\
                'De': de ,'De0': de - hardware_dict['max_split'], \
                'DE': dE ,'tgr': tgr , 'r1r': r1r , 'r2r': r2r }
            try:
                self.Simulate(super_dict,realistic = True )
                fidelity = self.fidelity
                p_success = self.p_success
                gate_time = self.gate_time
                
                cost =  gate_performance_cost_function(fidelity,p_success,gate_time)
            except:
                # If there was an error above, some variable went to infinity due to very small denominator.
                # As a result, the cost function should be very large in that case.
                cost = maximum_cost
            print(f'Cost function value: {cost} , fidelity={np.round(fidelity,decimals = 5)} , p_success={np.round(p_success,decimals=5)}, tg={np.round(gate_time,decimals=5)}')
            return cost

        # Callback function to track status 
        step = [0]
        def callback(x, *args, **kwargs):
            step[0] += 1
            print(f'step = {step[0]}, (De,DE,tgr,r1,r2) = {x} ')
        

        result = minimize(cost_function, initial_params , method = 'Nelder-Mead',options={'maxiter':max_iter,'disp': True},callback=callback)

        # Optimized parameters
        self.De_opt , self.DE_opt , self.tgr_opt, self.r1r_opt ,self.r2r_opt = tuple(result.x)  
        
        super_dict = {'C' : hardware_dict['C'] ,'c':hardware_dict['c'],\
                'De': self.De_opt ,'De0': self.De_opt - hardware_dict['max_split'], \
                'DE': self.DE_opt ,'tgr': self.tgr_opt , 'r1r': self.r1r_opt , 'r2r': self.r2r_opt}

        self.Simulate(super_dict , realistic = True)
        # Optimized performance
        ## Fidelity
        self.fidelity_opt =  self.fidelity
        ## Probability of success
        self.p_success_opt = self.p_success
        ## Gate time
        self.gate_time_opt = self.gate_time

        # Create a dictionary for easier examination
        self.optimized_dict = dict()
        optimized_parameters = ['fidelity','p_success' , 'gate_time', 'tgr', 'De', 'DE','r1r','r2r' ]
        for param in optimized_parameters:
            exec(f"self.optimized_dict['{param}'] = self.{param}_opt ")

        self.performance = {'fidelity': self.fidelity_opt , 'p_success': self.p_success_opt , 'gate_time': self.gate_time_opt}




class Analytical():
    '''
    Class designed for analytical Simulations.
    For this to be valid, gamma_g has to be 0.
    '''
        
    def __init__(self, SimulationClass):
        self.setup = SimulationClass.setup
        self.parameters = SimulationClass.parameters
        self.variables = SimulationClass.variables
        self.obtain_hamiltonian_lindblads()



    
    def obtain_hamiltonian_lindblads(self):
        '''
        Substitutes the general parameter regime in the hamiltonian and lindblads and organizes the lindblads.
        '''
        
        
        #Hamiltonian

        self.eff_hamiltonian_gs = [dictionary_substitution( self.setup.eff_hamiltonian_gs[diag,diag] , self.parameters ) 
                                    for diag in range(self.setup.gs_dim)]
        
        # Lindblad operators subroutine
        self.lind_op_number = len(self.setup.lindblad_list)
        self.eff_lind = []
        self.EffLindbladElements = []
        self.eff_lind_coeff = []
        for lind_op in range(self.lind_op_number ):
            self.eff_lind.append([])
            L_matrix = self.setup.eff_lindblad_list[lind_op]
            L_nonzeros = []
            L_nonzeros_pos = []
            AffectedStates = []
            for i in  range(L_matrix.nrows()):
                for j in  range(L_matrix.ncols()):
                    if  str(L_matrix[i,j]) != '0' and str(L_matrix[i,j]) != '0.0' :
                        L_nonzeros.append(L_matrix[i,j])
                        L_nonzeros_pos.append((i,j))
                        AffectedGSState = self.setup.pos_gs.index(j)
                        AffectedStates.append(AffectedGSState)

            self.eff_lind_coeff.append(self.setup.L_coeffs[lind_op])
            

            L_meq = [0 for i in range(self.setup.gs_dim ) ]
            for (i,state) in enumerate(AffectedStates):
                L_meq[state] += dictionary_substitution(L_nonzeros[i],self.parameters)
             
                    
            self.EffLindbladElements.append(L_meq)  


            



    def obtain_gate_performance_hardware(self, C_val , c_val, max_split):
        '''
        Obtain the gate performance given the hardware setting. Detunings are still allowed to vary.
        '''
        cooperativities_dict = {'C' : C_val ,'c': c_val }
        
        DE = sg.var('DE')
        detunings_dict_p = {'De':  DE + max_split/2 ,'De0':  DE - max_split/2,
                            'DE':  DE,  'dc': DE ,'db':  DE }
        
        positive_unitary_dict = {**cooperativities_dict , ** detunings_dict_p}

        GSRange = range(self.setup.gs_dim)

        H = [ dictionary_substitution( self.eff_hamiltonian_gs[diag] , positive_unitary_dict ) 
              for diag in GSRange]
        
        H_tot =[H[0] + H[1]+H[2] -H[3],
                H[0] + H[3],
                H[0] + H[3],
                H[3] + H[1]+H[2] -H[0]]

        total_gate_time =  2 * sg.abs_symbolic( np.pi  /(H[3]+H[0]-H[1]-H[2]) ) * sg.var('tgr') #tgr : gate time ratio
        
        total_gate_time_symbolic = sg.var('tgs')

        EffEffHamiltonian = np.array([0 for i in GSRange ])

        LossFactors = np.array([0 for i in GSRange])
        
        
        for lind_op in range(self.lind_op_number):
            for which in GSRange:
                
                L_num = self.EffLindbladElements[lind_op][which]
                if type(L_num) is type(sg.var('x')): 
                    L_num = dictionary_substitution(L_num , positive_unitary_dict)
                
                Loss = L_num* sg.conjugate(L_num)

                if str(self.setup.L_coeffs[lind_op])=='sqrt(gamma_g)':
                    #model loss of fidelity
                    pass
                else:
                    LossFactors = LossFactors + Loss * total_gate_time_symbolic / 4

        self.LossPerStateC = sg.vector(LossFactors)
        
        init_state = np.array([1,1,1,1])/2

        PSuccess = 0
        for i in GSRange:
            PSuccess += init_state[i]**2 * np.exp( - LossFactors[i])

        PSuccess_symbolic = sg.var('pss')


        PureEvolutionVector = []   
        for i in GSRange:
            Evolution = sg.exp(-1j* H_tot[i] * total_gate_time_symbolic)  
            PureEvolutionVector.append( Evolution*init_state[i] )

        PureEvolutionVectorSg = sg.vector(PureEvolutionVector) / sg.sqrt(PSuccess_symbolic)
        
        self.PureEvolutionVectorSg = PureEvolutionVectorSg

        # Post process rotations
        r1r = sg.var('r1r')
        r1 = - total_gate_time_symbolic/4 * (H_tot[0] - H_tot[2]) * r1r
        self.r1 = - total_gate_time_symbolic/4 * (H_tot[0] - H_tot[2])   
        dim = self.setup.gs_dim
        R1 = sg.Matrix(sg.SR,np.zeros((dim,dim)))
        R1_list = [sg.exp(sg.I*r1*i) for i in range(2) for j in range(2) ]
        for i in range(dim):
            R1[i,i] = R1_list[i]

        r2r = sg.var('r2r')
        r2 = -total_gate_time_symbolic/4  * (H[0] - H[1]) * r2r
        self.r2 = -total_gate_time_symbolic/4  * (H[0] - H[1])
        R2 = sg.Matrix(sg.SR,np.zeros((dim,dim)))
        R2_list = [sg.exp(sg.I*r2*j) for i in range(2) for j in range(2) ]
        for i in range(dim):    
            R2[i,i] = R2_list[i]

        # Simulate

        RotatedFinalVector = R1*R2*PureEvolutionVectorSg

        FinalVector = sg.vector(np.array([1,1,1,-1])/2)
        
        # The following expressions are not fully expressed in terms of detunings necessarily.
        self.fidelity_hw  = sg.abs_symbolic(RotatedFinalVector*FinalVector) # Contains tgs and pss
        self.p_success_hw       = PSuccess                                        # Contains tgs 
        self.gate_time_hw       = total_gate_time


        
        # The following expressions are fully expressed in terms of detunings
        self.gate_time_hw_full = self.gate_time_hw
        self.p_success_hw_full = self.p_success_hw.subs(tgs = self.gate_time_hw_full)
        self.fidelity_hw_full = self.fidelity_hw.subs(tgs = self.gate_time_hw_full , pss = self.p_success_hw_full )
        #self.concurrence_hw_full = self.concurrence_hw.subs(tgs = self.gate_time_hw_full , pss = self.p_success_hw_full )

        # A dictionary of the hardware setting
        self.hardware_dict = {'C' : C_val , 'c' : c_val , 'max_split' : max_split}

    
    def optimize_gate_performance_hardware(self, initial_params):
        '''
        Optimize the gate performance, minimizing some cost function.
        Initial parameters is the starting point for minimization.

        initial_params = [ De_in, DE_in, tgr_in ]
        '''
        
        #gate_time_fun = self.gate_time_hw.function(sg.var('tgr'),sg.var('De'),sg.var('DE'),sg.var('r1r'),sg.var('r2r')) #turn symbolic expression to function
        #p_success_fun = self.p_success_hw.function(sg.var('tgr'),sg.var('De'),sg.var('DE'),sg.var('r1r'),sg.var('r2r'),sg.var('tgs')) #turn symbolic expression to function
        #fidelity_fun  = self.fidelity_hw.function(sg.var('tgr'),sg.var('De'),sg.var('DE'),sg.var('r1r'),sg.var('r2r'),sg.var('tgs'),sg.var('pss'))   #turn symbolic expression to function
        gate_time_fun = self.gate_time_hw.subs(tgr=1).function(sg.var('DE')) #turn symbolic expression to function
        p_success_fun = self.p_success_hw.subs(tgr=1).function(sg.var('DE'),sg.var('tgs')) #turn symbolic expression to function
        #fidelity_fun  = self.fidelity_hw.function(sg.var('tgr'),sg.var('De'),sg.var('DE'),sg.var('r1r'),sg.var('r2r'),sg.var('tgs'),sg.var('pss'))   #turn symbolic expression to function
 
        

        def cost_function(params):
            '''
            Function that is used for minimization. Essentially it only makes use of global gate performance function.
            '''
            dE = params
            try:
                gate_time = sg.real(gate_time_fun(dE))
                p_success = sg.real(p_success_fun(dE,gate_time))
                #fidelity  = sg.real(fidelity_fun(t,de,dE, r1 , r2,gate_time,p_success))
                print(gate_time/p_success)
                return gate_time/p_success#gate_performance_cost_function(fidelity,p_success,gate_time)

            except:
                # If there was an error above, some variable went to infinity due to very small denominator.
                # As a result, the cost function should be very large in that case.
                return 10**30
        
        result = minimize(cost_function, initial_params , method = 'COBYLA')

        # Optimized parameters
        self.DE_opt = float(result.x)#tuple(result.x)    
        
        # Optimized performance
        ## Fidelity
        #self.fidelity_opt =  self.fidelity_hw_full.subs( tgr = self.tgr_opt , De = self.De_opt , DE = self.DE_opt,r1r=self.r1r_opt,r2r=self.r1r_opt )
        #self.fidelity_opt = float( sg.real( self.fidelity_opt.n())) # obtain number to be used later on efficiently
        ## Probability of success
        self.p_success_opt = self.p_success_hw_full.subs( tgr = 1 , DE = self.DE_opt )
        self.p_success_opt = float( sg.real( self.p_success_opt.n())) # obtain number to be used later on efficiently
        ## Gate time
        self.gate_time_opt = self.gate_time_hw_full.subs( tgr = 1 , DE = self.DE_opt  )
        self.gate_time_opt = float( sg.real( self.gate_time_opt.n())) # obtain number to be used later on efficiently

        # Create a dictionary for easier examination
        self.optimized_dict = dict()
        optimized_parameters = ['p_success' , 'gate_time',  'DE', ]
        for param in optimized_parameters:
            exec(f"self.optimized_dict['{param}'] = self.{param}_opt ")

        #self.performance = {'fidelity': self.fidelity_opt , 'p_success': self.p_success_opt , 'gate_time': self.gate_time_opt}