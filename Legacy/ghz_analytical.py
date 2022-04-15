from . import gate_simulation_functions 
import scipy.optimize 
import numpy as np
import sage.all as sg



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
        self.obtain_gate_performance_symbolic()

    
    def basic_substitution(self, symoblic_expression):   
        '''
        Takes a symbolic expression and substitutes the variables from the parameter dictionary, returning the result.
        '''
        q = sg.copy(symoblic_expression) 
        return q.subs(self.parameters)


    
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
                    if  str(L_matrix[i,j]) != '0' and str(L_matrix[i,j]) != '0.0' :
                        L_nonzeros.append(L_matrix[i,j])
                        L_nonzeros_pos.append((i,j))
                        AffectedGSState = self.setup.pos_gs.index(j)
                        AffectedStates.append(AffectedGSState)

            self.eff_lind_coeff.append(self.setup.L_coeffs[lind_op])
            

            L_meq = [0 for i in range(self.setup.gs_dim ) ]
            for (i,state) in enumerate(AffectedStates):
                L_meq[state] += self.basic_substitution(L_nonzeros[i])
             
                    
            self.EffLindbladElements.append(L_meq)       
            

    def obtain_gate_performance_symbolic(self):
        '''
        Obtain the performance of the gate when substituting only the self.variables list of variables with self.parameters
        '''
        
        GSRange = range(self.setup.gs_dim)


        H_symbolic = [ sg.var(f'H_{diag}') for diag in GSRange] 
        self.H_symbolic = H_symbolic
        
        self.gate_time_symbolic =  sg.abs_symbolic( np.pi  /(H_symbolic[3]+H_symbolic[0]-H_symbolic[1]-H_symbolic[2]) ) * sg.var('tgr') #tgr : gate time ratio
        gate_time_symbolic = sg.var('tgs')


        eff_eff_hamiltonian_symbolic = [-1j*H_symbolic[i] for i in GSRange ]

        L_symbolic = []
        loss_factors_symbolic = [0 for i in GSRange]
        for lind_op in range(self.lind_op_number):
            for which in GSRange:   
                l_symbolic =  sg.var(sg.var(f'L_{lind_op}_{which}' , domain='complex'))     
                L_symbolic.append(l_symbolic)     
                loss_symbolic = l_symbolic* sg.conjugate(l_symbolic)
                eff_eff_hamiltonian_symbolic[which]  -= loss_symbolic / 2
                loss_factors_symbolic[which] += loss_symbolic * gate_time_symbolic
        self.L_symbolic = L_symbolic
        

        init_state = np.array([1,1,1,1])/2

        p_success_symbolic = 0
        for i in GSRange:
            p_success_symbolic += init_state[i]**2 * np.exp( - loss_factors_symbolic[i])
        self.p_success_symbolic = p_success_symbolic
        
        PSuccess_symbolic = sg.var('pss')

        pure_evolution_unitary_symbolic = []  
        for i in GSRange:
            evolution_symbolic = sg.exp(eff_eff_hamiltonian_symbolic[i]*gate_time_symbolic)  
            pure_evolution_unitary_symbolic.append( evolution_symbolic )

        evolution_unitary_symbolic = sg.Matrix(sg.SR,np.diag(pure_evolution_unitary_symbolic))/ sg.sqrt(PSuccess_symbolic)
        
        # Post process rotations
        rotation = - gate_time_symbolic * (H_symbolic[0] - H_symbolic[2]) 

        # The following expressions are not fully expressed in terms of detunings necessarily.
        self.fidelity_ghz3_symbolic   = gate_simulation_functions.GHZ_3_symbolic_fidelity_from_evolution_4rots(evolution_unitary_symbolic,rotation)




    def obtain_gate_performance_hardware(self, C_val , c_val, max_split):
        '''
        Obtain the gate performance given the hardware setting. Detunings are still allowed to vary.
        '''
        
        De0_val =  sg.var('De') - max_split

        CcDe0_dict = {sg.var('C'): C_val , sg.var('c'): c_val ,sg.var('De0'): De0_val}

        self.HL_dict_hw = {}
        for diag,var in enumerate(self.H_symbolic):
            self.HL_dict_hw[var] = self.basic_substitution(self.setup.eff_hamiltonian_gs[diag,diag]).subs(CcDe0_dict) 
        
        for elem,var in enumerate(self.L_symbolic):
            lind_op = elem // 4
            which   = elem % 4
            L_num = self.EffLindbladElements[lind_op][which]
            if type(L_num) is type(sg.var('x')): L_num = L_num.subs(CcDe0_dict)
            self.HL_dict_hw[var] = L_num

        # A dictionary of the hardware setting
        self.hardware_dict = {'C' : C_val , 'c' : c_val , 'max_split' : max_split}


    def tunable_performance(self, tunable_parameters_dict):
        '''
        tunable_parameters = [ De_val, DE_val, tg_ratio , r1_ratio , r2_ratio, r3_ratio, r4_ratio ]
        '''
        try:
            HL_dict = self.HL_dict_hw.copy()
            for key in HL_dict:
                if type(HL_dict[key]) is type(sg.var('x')): HL_dict[key] = HL_dict[key].subs(tunable_parameters_dict)
            HL_dict = { **HL_dict ,**tunable_parameters_dict }
            gate_time = float(sg.real( self.gate_time_symbolic.subs( HL_dict )))
            HL_dict[sg.var('tgs')] =  gate_time
            tunable_parameters_dict[sg.var('tgs')] =  gate_time
            p_success = float(sg.real(self.p_success_symbolic.subs(HL_dict)))
        
            tunable_parameters_dict[sg.var('pss')]   = p_success
            HL_dict[sg.var('pss')]   = p_success
            fidelity_3ghz = float(sg.real(self.fidelity_ghz3_symbolic.subs( HL_dict)))

            return gate_time, p_success, fidelity_3ghz
        except ValueError:
            return 10**9 , 0 , 0

    
    
    def optimize_gate_performance_hardware(self, initial_params, fidelity_cap = 0.95 , interval_of_confidence = 0.99):
        '''
        Optimize the gate performance, minimizing some cost function.
        Initial parameters is the starting point for minimization.

        initial_params = [ De_val, DE_val, tg_ratio , r1_ratio , r2_ratio, r3_ratio, r4_ratio ]
        '''
        
        #gate_time_fun = self.gate_time_hw.function(sg.var('tgr'),sg.var('De'),sg.var('DE'),sg.var('r1r'),sg.var('r2r')) #turn symbolic expression to function
        #p_success_fun = self.p_success_hw.function(sg.var('tgr'),sg.var('De'),sg.var('DE'),sg.var('r1r'),sg.var('r2r'),sg.var('tgs')) #turn symbolic expression to function
        #fidelity_fun  = self.fidelity_hw.function(sg.var('tgr'),sg.var('De'),sg.var('DE'),sg.var('r1r'),sg.var('r2r'),sg.var('tgs'),sg.var('pss'))   #turn symbolic expression to function
        
        parameters_to_optimize = [ sg.var('De'),  sg.var('DE'), sg.var('tgr') , sg.var('r1_r') , sg.var('r2_r'), sg.var('r3_r'), sg.var('r4_r') ]
        num_parameters = len(parameters_to_optimize)
        
        if type(initial_params) == dict:
            initial_params = [initial_params[param] for param in parameters_to_optimize]
        
        
        def cost_function(params):
            '''
            Function that is used for minimization. Essentially it only makes use of global gate performance function.
            '''
            De_val, DE_val, tg_ratio , r1_ratio , r2_ratio, r3_ratio, r4_ratio = params
            
            tunable_parameters_dict = {sg.var('De') : De_val , sg.var('DE') : DE_val 
                    , sg.var('tgr'): tg_ratio
                    , sg.var('r1_r'):r1_ratio, sg.var('r2_r'):r2_ratio,
                        sg.var('r3_r'):r3_ratio,sg.var('r4_r'):r4_ratio}
            gate_time, p_success, fidelity_3ghz = self.tunable_performance(tunable_parameters_dict)
            
            return gate_simulation_functions.gate_performance_cost_function(fidelity_3ghz,p_success,gate_time, fidelity_cap, interval_of_confidence)
        
        
        result = minimize(cost_function, initial_params , method = 'Nelder-Mead')

        scipy.optimize.differential_evolution(cost_function,bounds=bounds,updating='deferred', workers=number_of_cores_to_use,disp=True)

        # Optimized parameters
        self.opt_tunable_dict = {parameters_to_optimize[i]: result.x[i] for i in range(num_parameters)}
            
        self.gate_time_opt, self.p_success_opt, self.fidelity_opt   = self.tunable_performance(self.opt_tunable_dict)

        # Create a dictionary for easier examination
        self.optimized_performance = {'fidelity': self.fidelity_opt , 'p_success': self.p_success_opt , 'gate_time': self.gate_time_opt}