#
# File containing all necessary classes to simulate the gate
#

from system import *
from qutip.qip.operations import rz
from scipy.optimize import minimize
import copy


def gate_performance_cost_function(fidelity,p_success,gate_time):
    '''
    Cost function to be minimized in order to achieve maximum performance.
    Designed to handle arrays as well.
    The function should be smaller than 1.
    '''

    def performance_function(fidelity,p_success,gate_time):
        '''
        Here the actual cost function is defined.
        '''
        fidelity_cap = 0.95
        avg_time_to_success = max(gate_time*p_success , 1e-10)
        if fidelity >= fidelity_cap:
            cost = 1 - fidelity_cap - 1/avg_time_to_success
        else:
            cost = 1 - fidelity
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



class Simulation():

    def __init__(self,setup_char):
        self.setup_char = setup_char
        self.setup = system(setup_char,MMA=True,ManyVariables=False,TwoPhotonResonance= True)
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
        gamma_val = 1
        gamma0_val = gamma_val
        
        # Used for analytical results
        gamma_g_val = 0 # DO NOT CHANGE
        gamma_f_val = 1 * gamma_val
        
        # Used for realistic results
        gamma_g_real = 0.1 * gamma_val
        gamma_f_real = 0.9 * gamma_val

        kappa_b_val = 100 * gamma_val
        kappa_c_val = 100 * gamma_val
        
        C = sg.var('C')
        g_val = sg.sqrt( C * kappa_c_val * gamma_val)
        
        Omega_val =  sg.sqrt(C) * gamma_val * 0.25 / 2

        phi_val = 0

        g_f_val = g_val
        g0_val  = g_val
        
        c = sg.var('c')
        v_val   = sg.sqrt( c * kappa_c_val * kappa_b_val)   

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
        sg.var('gamma','DE','De','g','g_f','Omega','v','gamma_f','gamma_g','gamma0','De0','phi','g0','gamma0',domain='positive')
        sg.var('r0',domain='real',latex_name =r'r_0')
        sg.var('R_f',domain='real')#ratio  (g_f/g)^2
        sg.var('R0',domain='positive',  latex_name =r'R_0')
        sg.var('R_v',domain='real',  latex_name =r'R_{\nu}') #ratio (v/g)^2
        sg.var('r_g',domain='real',latex_name =r'r_g')
        sg.var('r_b',domain='real')
        sg.var('r_f',domain='real',latex_name =r'r_f')
        sg.var('kappa_c','kappa_b',domain='positive')




class Analytical():
    '''
    Class designed for analytical Simulations.
    For this to be valid, gamma_g has to be 0.
    '''
        
    def __init__(self, SimulationClass):
        self.setup = SimulationClass.setup
        self.parameters = SimulationClass.parameters
        self.variables = SimulationClass.variables
        self.obtain_lindblads()
        self.obtain_gate_performance()

    
    def basic_substitution(self, symoblic_expression):   
        '''
        Takes a symbolic expression and substitutes the variables from the parameter dictionary, returning the result.
        '''

        q = sg.copy(symoblic_expression)
        for var in self.variables:
            q = eval(f"q.subs({var}= self.parameters['{var}'] )") 
        return q


    
    def obtain_lindblads(self):
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
                    if not L_matrix[i,j].is_zero():
                        L_nonzeros.append(L_matrix[i,j])
                        L_nonzeros_pos.append((i,j))
                        AffectedGSState = self.setup.pos_gs.index(j)
                        AffectedStates.append(AffectedGSState)

            self.eff_lind_coeff.append(self.setup.L_coeffs[lind_op])
            

            L_meq = [0 for i in range(self.setup.gs_dim ) ]
            for (i,state) in enumerate(AffectedStates):
                L_meq[state] += self.basic_substitution(L_nonzeros[i])
             
                    
            self.EffLindbladElements.append(L_meq)       
            
          


    def obtain_gate_performance(self):
        '''
        Obtain the performance of the gate when substituting only the self.variables list of variables with self.parameters
        '''

        GSRange = range(self.setup.gs_dim)

        H = [self.basic_substitution(self.setup.eff_hamiltonian_gs[diag,diag]) for diag in GSRange]


        gate_time =  sg.abs_symbolic( np.pi  /(H[3]+H[0]-H[1]-H[2]) ) * sg.var('tgr')  #tgr : gate time ratio
        
        # To shorten expression of Psuccess we use tgs instead of the full expression for time
        gate_time_symbolic = sg.var('tgs')

        EffEffHamiltonian = [-1j*H[i] for i in GSRange ]

        LossFactors = [0 for i in GSRange]
        

        for lind_op in range(self.lind_op_number):
            for which in GSRange:
                
                L_num = self.EffLindbladElements[lind_op][which]
                Loss = L_num* sg.conjugate(L_num)

                EffEffHamiltonian[which] -= Loss/2
                LossFactors[which] += Loss * gate_time_symbolic 

        self.LossPerState = sg.vector(LossFactors)

        init_state = np.array([1,1,1,1])/2

        PSuccess = 0
        for i in GSRange:
            PSuccess += init_state[i]**2 * np.exp( - LossFactors[i])

        # To shorten expression of Fidelity, we use tgs instead of the full expression for time and pss instead of PSuccess
        PSuccess_symbolic = sg.var('pss')


        PureEvolutionVector = []   
        for i in GSRange:
            Evolution = sg.exp(EffEffHamiltonian[i]*gate_time_symbolic)  
            PureEvolutionVector.append( Evolution*init_state[i] )

        PureEvolutionVectorSg = sg.vector(PureEvolutionVector) / sg.sqrt(PSuccess_symbolic)

        # Post process rotations
        r1 = - gate_time_symbolic * (H[0] - H[2])           
        dim = self.setup.gs_dim
        R1 = sg.Matrix(sg.SR,np.zeros((dim,dim)))
        R1_list = [sg.exp(sg.I*r1*i) for i in range(2) for j in range(2) ]
        for i in range(dim):
            R1[i,i] = R1_list[i]

        r2 = -gate_time_symbolic * (H[0] - H[1])
        R2 = sg.Matrix(sg.SR,np.zeros((dim,dim)))
        R2_list = [sg.exp(sg.I*r2*j) for i in range(2) for j in range(2) ]
        for i in range(dim):    
            R2[i,i] = R2_list[i]

        # Simulate

        RotatedFinalVector = R1*R2*PureEvolutionVectorSg

        FinalVector = sg.vector(np.array([1,1,1,-1])/2)

        self.fidelity   = sg.abs_symbolic(RotatedFinalVector*FinalVector)
        self.p_success  = PSuccess
        self.gate_time  = gate_time



    def obtain_gate_performance_hardware(self, C_val , c_val, max_split):
        '''
        Obtain the gate performance given the hardware setting. Detunings are still allowed to vary.
        '''
        
        De0_val =  sg.var('De') - max_split
        
        GSRange = range(self.setup.gs_dim)

        H = [self.basic_substitution(self.setup.eff_hamiltonian_gs[diag,diag]).subs(C=C_val, c=c_val, De0 = De0_val) for diag in GSRange]


        gate_time =  sg.abs_symbolic( np.pi  /(H[3]+H[0]-H[1]-H[2]) ) * sg.var('tgr') #tgr : gate time ratio
        
        gate_time_symbolic = sg.var('tgs')

        EffEffHamiltonian = [-1j*H[i] for i in GSRange ]

        LossFactors = [0 for i in GSRange]
        

        for lind_op in range(self.lind_op_number):
            for which in GSRange:
                
                L_num = self.EffLindbladElements[lind_op][which]
                if type(L_num) is type(sg.var('x')): L_num = L_num.subs(C=C_val, c=c_val, De0 = De0_val)
                
                Loss = L_num* sg.conjugate(L_num)

                EffEffHamiltonian[which] -= Loss/2
                LossFactors[which] += Loss * gate_time_symbolic 

        self.LossPerStateC = sg.vector(LossFactors)

        init_state = np.array([1,1,1,1])/2

        PSuccess = 0
        for i in GSRange:
            PSuccess += init_state[i]**2 * np.exp( - LossFactors[i])

        PSuccess_symbolic = sg.var('pss')


        PureEvolutionVector = []   
        for i in GSRange:
            Evolution = sg.exp(EffEffHamiltonian[i]*gate_time_symbolic)  
            PureEvolutionVector.append( Evolution*init_state[i] )

        PureEvolutionVectorSg = sg.vector(PureEvolutionVector) / sg.sqrt(PSuccess_symbolic)

        # Post process rotations
        r1 = - gate_time_symbolic * (H[0] - H[2])          
        dim = self.setup.gs_dim
        R1 = sg.Matrix(sg.SR,np.zeros((dim,dim)))
        R1_list = [sg.exp(sg.I*r1*i) for i in range(2) for j in range(2) ]
        for i in range(dim):
            R1[i,i] = R1_list[i]

        r2 = -gate_time_symbolic * (H[0] - H[1])
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
        self.gate_time_hw       = gate_time
        
        # The following expressions are fully expressed in terms of detunings
        self.gate_time_hw_full = self.gate_time_hw
        self.p_success_hw_full = self.p_success_hw.subs(tgs = self.gate_time_hw_full)
        self.fidelity_hw_full = self.fidelity_hw.subs(tgs = self.gate_time_hw_full , pss = self.p_success_hw_full )

        # A dictionary of the hardware setting

        self.hardware_dict = {'C' : C_val , 'c' : c_val , 'max_split' : max_split}

    
    def optimize_gate_performance_hardware(self, initial_params):
        '''
        Optimize the gate performance, minimizing some cost function.
        Initial parameters is the starting point for minimization.

        initial_params = [ De_in, DE_in, tgr_in ]
        '''
        
        fidelity_fun  = self.fidelity_hw_full.function(sg.var('tgr'),sg.var('De'),sg.var('DE'))   #turn symbolic expression to function
        p_success_fun = self.p_success_hw_full.function(sg.var('tgr'),sg.var('De'),sg.var('DE')) #turn symbolic expression to function
        gate_time_fun = self.gate_time_hw_full.function(sg.var('tgr'),sg.var('De'),sg.var('DE')) #turn symbolic expression to function

        def cost_function(params):
            '''
            Function that is used for minimization. Essentially it only makes use of global gate performance function.
            '''
            de, dE, t = params
            try:
                fidelity  = sg.real(fidelity_fun(t,de,dE))
                p_success = sg.real(p_success_fun(t,de,dE))
                gate_time = sg.real(gate_time_fun(t,de,dE))
                if performance_makes_sense(fidelity,p_success,gate_time):
                    return gate_performance_cost_function(fidelity,p_success,gate_time)
                else:
                    return 1
            except:
                # If there was an error above, some variable went to infinity due to very small denominator.
                # As a result, the cost function should be very large in that case.
                return 1
        
        result = minimize(cost_function, initial_params , method = 'Nelder-Mead')

        # Optimized parameters
        self.De_opt , self.DE_opt , self.tgr_opt = tuple(result.x)    
        
        # Optimized performance
        ## Fidelity
        self.fidelity_opt =  self.fidelity_hw_full.subs( tgr = self.tgr_opt , De = self.De_opt , DE = self.DE_opt )
        self.fidelity_opt = float( sg.real( self.fidelity_opt.n())) # obtain number to be used later on efficiently
        ## Probability of success
        self.p_success_opt = self.p_success_hw_full.subs( tgr = self.tgr_opt , De = self.De_opt , DE = self.DE_opt )
        self.p_success_opt = float( sg.real( self.p_success_opt.n())) # obtain number to be used later on efficiently
        ## Gate time
        self.gate_time_opt = self.gate_time_hw_full.subs( tgr = self.tgr_opt , De = self.De_opt , DE = self.DE_opt )
        self.gate_time_opt = float( sg.real( self.gate_time_opt.n())) # obtain number to be used later on efficiently

        # Create a dictionary for easier examination
        self.optimized_dict = dict()
        optimized_parameters = ['fidelity','p_success' , 'gate_time', 'tgr', 'De', 'DE' ]
        for param in optimized_parameters:
            exec(f"self.optimized_dict['{param}'] = self.{param}_opt ")

        self.performance = {'fidelity': self.fidelity_opt , 'p_success': self.p_success_opt , 'gate_time': self.gate_time_opt}


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
        def substitute(a , realistic):
            q = sg.copy(a)
            if realistic:
                for var in self.variables:
                    q = eval(f"q.subs({var}= self.realistic_parameters['{var}'] )") 
            else:
                for var in self.variables:
                    q = eval(f"q.subs({var}= self.parameters['{var}'] )") 
            return q
            
        self.eff_hamiltonian_C =  substitute(self.eff_hamiltonian, realistic) 

        self.eff_lind_master_eq_C = []
        for lind_op in range(self.lind_op_number):
            self.eff_lind_master_eq_C.append(  substitute(self.eff_lind_master_eq[lind_op] , realistic )     )


    def Simulate(self, super_dict , realistic):
        '''
        Run simulation for a specified set of values.

        super_dict = dict['C','c','De','De0','DE','tgr']

        realistic : boolean
        '''
        
        self.basic_substitution(realistic)
        
        super_variables = ['C','c','De','De0','DE','tgr']
        self.super_dict = super_dict
        
        def evaluate_numerically(a, super_variables = super_variables, super_dict = super_dict):
            '''
            Substitutes all the values 
            '''
            q = sg.copy(a)
            for var in super_variables:
                q = eval(f"q.subs({var}= super_dict['{var}'] )") 
            return q
            
        
        eff_hamiltonian_num = evaluate_numerically(self.eff_hamiltonian_C).expand() 
        eff_hamiltonian_num = eff_hamiltonian_num.numpy().astype(float)
        H_obj = qt.Qobj(eff_hamiltonian_num)

        self.H_obj = H_obj

        #Initialize superoperator with hamiltonian
        L = -1j*( qt.spre(H_obj) -qt.spost(H_obj))

        L_obj_list = []
        L_np_list = []
        for lind_op in range(len(self.eff_lind_master_eq_C)):
            
            L_nparray = evaluate_numerically( self.eff_lind_master_eq_C[lind_op] ).numpy().astype(complex)
            L_np_list.append(L_nparray)
            L_obj_list.append(qt.Qobj(L_nparray))
            
            lind = qt.Qobj(L_nparray)
            L += qt.to_super(lind) - 0.5 * (  qt.spre(lind.dag()*lind) + qt.spost(lind.dag()*lind) )
        
        self.L_obj_list = L_obj_list
        
        gs_pos = self.GsPosInGsDec

        init_state = np.zeros(self.setup.gs_dim +self.setup.dec_dim)
        init_state[gs_pos] = 1/2 # |++> state
        psi0 = qt.Qobj(init_state)
        init_dm = qt.ket2dm(psi0)
        init_vec_dm = qt.operator_to_vector(init_dm)

        psif_gs  =  qt.Qobj(np.array([1,1,1,-1])/2)  #target state

        H = [eff_hamiltonian_num[gs_pos[0],gs_pos[0]], eff_hamiltonian_num[gs_pos[1],gs_pos[1]]\
            ,eff_hamiltonian_num[gs_pos[2],gs_pos[2]], eff_hamiltonian_num[gs_pos[3],gs_pos[3]]]
        
        gate_time =  np.abs( np.pi  /(H[3]+H[0]-H[1]-H[2]) ) * super_dict['tgr'] 

        
        if np.isposinf(gate_time) or np.isneginf(gate_time):
            return  0,1 , 1
        Lt = (L*gate_time).expm()
        #post process rotations
        
        rz = qt.qip.operations.rz

        r1 = - gate_time * (H[0] - H[2])           
        R1 = qt.Qobj(qt.tensor( rz(r1 ),qt.identity(2)).full())
        
        
        r2 = -gate_time * (H[0] - H[1])
        R2 = qt.Qobj( qt.tensor(qt.identity(2),rz(r2 )).full())
        
        #simulate
        dm_f_vec = Lt*init_vec_dm

        dm_f = qt.vector_to_operator(dm_f_vec)


        dec_pos = self.DecPosInGsDec
        
        Psuccess = 1
        for pos in dec_pos:
            Psuccess -= dm_f[pos,pos]
        p_success = np.real(Psuccess)

        dm_f_gs =   1/ Psuccess * dm_f.eliminate_states(dec_pos) #ground state
        
        RotatedFinalState = R1.dag() * R2.dag() * psif_gs
        
        fidelity = qt.fidelity(dm_f_gs,RotatedFinalState)

        self.fidelity = fidelity
        self.p_success = p_success
        self.gate_time = gate_time

        performance = {'fidelity': fidelity , 'p_success': p_success , 'gate_time': gate_time}
        self.performance = {'fidelity': self.fidelity , 'p_success': self.p_success , 'gate_time': self.gate_time}

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
            de, dE, tgr = params
            super_dict = {'C' : hardware_dict['C'] ,'c':hardware_dict['c'],\
                'De': de ,'De0': de - hardware_dict['max_split'], \
                'DE': dE ,'tgr': tgr}
            try:
                self.Simulate(super_dict,realistic = True )
                fidelity = self.fidelity
                p_success = self.p_success
                gate_time = self.gate_time
                if performance_makes_sense(fidelity,p_success,gate_time):
                    cost =  gate_performance_cost_function(fidelity,p_success,gate_time)
                else:
                    cost = 1
            except:
                # If there was an error above, some variable went to infinity due to very small denominator.
                # As a result, the cost function should be very large in that case.
                cost = 1
            print(f'Cost function value: {cost} , fidelity={fidelity} , p_success={p_success}')
            return cost

        # Callback function to track status 
        step = [0]
        def callback(x, *args, **kwargs):
            step[0] += 1
            print(f'step = {step[0]}, (De,DE,tgr) = {x} ')
        

        result = minimize(cost_function, initial_params , method = 'Nelder-Mead',options={'maxiter':max_iter,'disp': True},callback=callback)

        # Optimized parameters
        self.De_opt , self.DE_opt , self.tgr_opt = tuple(result.x)  
        
        super_dict = {'C' : hardware_dict['C'] ,'c':hardware_dict['c'],\
                'De': self.De_opt ,'De0': self.De_opt - hardware_dict['max_split'], \
                'DE': self.DE_opt ,'tgr': self.tgr_opt}

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
        optimized_parameters = ['fidelity','p_success' , 'gate_time', 'tgr', 'De', 'DE' ]
        for param in optimized_parameters:
            exec(f"self.optimized_dict['{param}'] = self.{param}_opt ")

        self.performance = {'fidelity': self.fidelity_opt , 'p_success': self.p_success_opt , 'gate_time': self.gate_time_opt}