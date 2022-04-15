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
        self.fidelity_ghz4_symbolic   = gate_simulation_functions.GHZ_4_symbolic_fidelity_from_evolution(evolution_unitary_symbolic,rotation)

        self.fidelity_ghz3_symbolic   = gate_simulation_functions.GHZ_3_symbolic_fidelity_from_evolution(evolution_unitary_symbolic,rotation)




    def obtain_gate_performance_hardware(self, hardware_dict):
        '''
        Obtain the gate performance given the hardware setting. Detunings are still allowed to vary.
        '''
        
        De0_val =  sg.var('De') - hardware_dict['max_split']

        CcDe0_dict = {sg.var('C'): hardware_dict['C'] , sg.var('c'): hardware_dict['c'] ,sg.var('De0'): De0_val}

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
        self.hardware_dict = hardware_dict


    def tunable_performance(self, tunable_parameters_dict,opt_settings_dict):
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
            if    opt_settings_dict["ghz_dim"] == 4:
                fidelity = float(sg.real(self.fidelity_ghz4_symbolic.subs( HL_dict)))
            elif  opt_settings_dict["ghz_dim"] == 3:
                fidelity = float(sg.real(self.fidelity_ghz3_symbolic.subs( HL_dict)))
            
        except ValueError:
            gate_time  =  10**9  
            fidelity = 0
            p_success = 0

        performance_dict = {'gate_time': gate_time ,'p_success':p_success, 'fidelity': fidelity}

        return performance_dict

    
    
    def optimize_gate_performance_hardware(self, parameter_bounds, opt_settings_dict ):
        '''
        Optimize the gate performance, minimizing some cost function.
        parameter_bounds is the starting point for minimization.

        initial_params = [ De_val, DE_val, tg_ratio , r0_ratio, r1_ratio , r2_ratio, ... ]
        '''
        
        parameters_to_optimize = [ sg.var('De'),  sg.var('DE'), sg.var('tgr')]
        # Add rotation parameters depending on the ghz_dim
        #for i in range(opt_settings_dict["ghz_dim"] ): parameters_to_optimize.append( sg.var(f'r{i}_r') ) 
        for i in range(4): parameters_to_optimize.append( sg.var(f'r{i}_r') ) 
        num_parameters = len(parameters_to_optimize)    
        
        global cost_function
        def cost_function(params):
            '''
            Function that is used for minimization. Essentially it only makes use of global gate performance function.
            '''
            # Obtain non rotational parameters
            De_val, DE_val, tg_ratio  = params[0:3]
            tunable_parameters_dict = {sg.var('De') : De_val , sg.var('DE') : DE_val , sg.var('tgr'): tg_ratio}
            
            for i,rot_val in enumerate(params[3:]):
                tunable_parameters_dict[sg.var(f'r{i}_r')] = rot_val
                    
            performance_dict = self.tunable_performance(tunable_parameters_dict,opt_settings_dict)
            
            tunable_parameters_dict.clear()
            return gate_simulation_functions.gate_performance_cost_function(performance_dict, opt_settings_dict)
        
        result = scipy.optimize.differential_evolution(cost_function,
                bounds=parameter_bounds,updating='deferred', workers=opt_settings_dict["n_cores"],disp=opt_settings_dict["disp_bool"])

        # Optimized parameters
        self.opt_tunable_dict = {parameters_to_optimize[i]: result.x[i] for i in range(num_parameters)}
            
        self.optimized_performance_dict = self.tunable_performance(self.opt_tunable_dict,opt_settings_dict)

        self.optimized_performance_dict['t_conf'] = gate_simulation_functions.time_interval_of_confidence(opt_settings_dict,self.optimized_performance_dict)
