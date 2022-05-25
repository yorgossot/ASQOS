from . import gate_simulation_functions 
import scipy.optimize 
import numpy as np
import qutip as qt
import sage.all as sg
from functools import partial
from ...notebook_library import MMA_simplify
import pickle


class Analytical():
    '''
    Class designed for analytical Simulations.
    For this to be valid, gamma_g has to be 0.
    '''
        
    def __init__(self, SimulationClass):
        self.setup = SimulationClass.setup
        self.SimulationClass = SimulationClass
        if SimulationClass.setup_char[-1] == SimulationClass.setup_char[0]:
            self.symmetric = True  # assuming that setup is symmetric
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
        self.GSRange = GSRange

        H_symbolic = [ sg.var(f'H_{diag}') for diag in GSRange] 
        self.H_symbolic = H_symbolic
        
        self.gate_time_symbolic =  sg.abs_symbolic( np.pi  /(H_symbolic[3]+H_symbolic[0]-H_symbolic[1]-H_symbolic[2]) ) * sg.var('tgr') #tgr : gate time ratio
        gate_time_symbolic = sg.var('tgs')


        eff_eff_hamiltonian_symbolic = [H_symbolic[i] for i in GSRange ]

        L_symbolic = []
        loss_factors_symbolic = [0 for i in GSRange]
        for lind_op in range(self.lind_op_number):
            for which in GSRange:   
                l_symbolic =  sg.var(sg.var(f'L_{lind_op}_{which}' , domain='complex'))     
                L_symbolic.append(l_symbolic)     
                loss_symbolic = l_symbolic* sg.conjugate(l_symbolic)
                eff_eff_hamiltonian_symbolic[which]  -= sg.I*loss_symbolic / 2
                loss_factors_symbolic[which] += loss_symbolic * gate_time_symbolic
        self.L_symbolic = L_symbolic
        
        n_qubits = 2  # 2 qubit case
        plus_state = qt.Qobj(np.array([1,1])/np.sqrt(2) )
        init_state = qt.tensor(*(plus_state for i in range(n_qubits)))  

        # Start at ++
        init_state = sg.vector(init_state.data.toarray().reshape(2**n_qubits)).column()

        # Apply initial adjustments
        for i in range(2):
            rot_val = sg.var(f'r0_i') 
            R_matr = gate_simulation_functions.R_y(rot_val)
            init_state = gate_simulation_functions.ten_r(R_matr,i,n_qubits )* init_state


        p_success_symbolic = 0
        for i in GSRange:
            p_success_symbolic += sg.abs_symbolic(init_state[i,0])**2 * np.exp( - loss_factors_symbolic[i])
        self.p_success_symbolic = p_success_symbolic
        
        PSuccess_symbolic = sg.var('pss')

        pure_evolution_symbolic = []  
        for i in GSRange:
            evolution_symbolic = sg.exp(- sg.I * eff_eff_hamiltonian_symbolic[i]*gate_time_symbolic)  
            pure_evolution_symbolic.append( evolution_symbolic )

        evolution_symbolic_HL = sg.Matrix(sg.SR,np.diag(pure_evolution_symbolic))
        
        self.evolution_symbolic_HL = evolution_symbolic_HL

        evolution_symbols = [sg.var(f'u0'),sg.var(f'u1'),sg.var(f'u1'),sg.var(f'u2') ] 
        evolution_symbolic = sg.Matrix(sg.SR,np.diag(evolution_symbols))
        self.evolution_symbolic = evolution_symbolic
        # Post process rotations
        self.rotation_symbolic_HL =  gate_time_symbolic * (H_symbolic[0] - H_symbolic[2]) 
        self.rotation_symbolic = sg.var('rot')

        # The following expressions are not fully expressed in terms of detunings necessarily.
        
        ghz_dims = [2]
        self.GHZ_obtain_symbolic_fidelity_from_evolution(ghz_dims) 





    def obtain_gate_performance_hardware(self, hardware_dict):
        '''
        Obtain the gate performance given the hardware setting. Detunings are still allowed to vary.
        '''
        # A dictionary of the hardware setting
        self.hardware_dict = hardware_dict

        De0_val =  sg.var('De') - hardware_dict[sg.var('D_max')]
        
        # Contains parametric De0, thus not added to self.hardware_dict
        hw_subs_dict = {**hardware_dict, sg.var('De0'):De0_val}
        

        self.substitution_dict_hw = {}
        for diag,var in enumerate(self.H_symbolic):
            self.substitution_dict_hw[var] = self.basic_substitution(self.setup.eff_hamiltonian_gs[diag,diag]).subs(hw_subs_dict) 
        
        for elem,var in enumerate(self.L_symbolic):
            lind_op = elem // 4
            which   = elem % 4
            L_num = self.EffLindbladElements[lind_op][which]
            if type(L_num) is type(sg.var('x')): L_num = L_num.subs(hw_subs_dict)
            self.substitution_dict_hw[var] = L_num
        
        # Obtain unitary in terms of hardware parameters
        self.hardware_evolution_unitary = self.evolution_symbolic_HL.subs(self.substitution_dict_hw)

        self.substitution_dict_hw[sg.var(f'u0')] = self.hardware_evolution_unitary[0,0]
        self.substitution_dict_hw[sg.var(f'u1')] = self.hardware_evolution_unitary[1,1]
        self.substitution_dict_hw[sg.var(f'u2')] = self.hardware_evolution_unitary[3,3]
        
        self.rotation_hardware = self.rotation_symbolic_HL(self.substitution_dict_hw)

        self.substitution_dict_hw[sg.var('rot')] = self.rotation_hardware




    def tunable_performance(self, tunable_parameters_dict,opt_settings_dict):
        '''
        tunable_parameters = [ De_val, DE_val, tg_ratio , r1_ratio , r2_ratio, r3_ratio, r4_ratio ]
        '''
        try:
            tunable_parameters_dict[sg.var('DE')] = tunable_parameters_dict[sg.var('DE_C')] * tunable_parameters_dict[sg.var('C')]
            HL_dict = self.substitution_dict_hw.copy()
            for key in HL_dict:
                if type(HL_dict[key]) is type(sg.var('x')): HL_dict[key] = HL_dict[key].subs(tunable_parameters_dict)

            HL_dict = { **HL_dict ,**tunable_parameters_dict }


            gate_time = float(sg.real( self.gate_time_symbolic.subs( HL_dict )))
            HL_dict[sg.var('tgs')] =  gate_time

            p_success = float(sg.real(self.p_success_symbolic.subs(HL_dict)))
            HL_dict[sg.var('pss')]   = p_success

            aux_dict = self.GHZ_fidelity_dict[opt_settings_dict['ghz_dim']]['aux_dict'].copy()
            for key in aux_dict:
                HL_dict[key] = aux_dict[key].subs(HL_dict).subs(tgs=gate_time)
            
            for i in range(3):
                HL_dict[sg.var(f'u{i}')] = HL_dict[sg.var(f'u{i}')].subs(HL_dict)

            
            HL_dict[sg.var('rot')] = self.rotation_hardware.subs(HL_dict)
            
            fid_expression = self.GHZ_fidelity_dict[opt_settings_dict["ghz_dim"]]['fidelity']
               
            fidelity =   float(sg.real(fid_expression.subs(HL_dict).subs( gs=gate_time)))

            HL_dict.clear()        
            
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
        
        parameters_to_optimize = [sg.var('C'),sg.var('De'),  sg.var('DE_C'), sg.var('tgr')]
        num_of_non_rot_params = len(parameters_to_optimize)
        # Add rotation parameters depending on the ghz_dim

        if opt_settings_dict["ghz_dim"] > 2:
            for i in range(opt_settings_dict["ghz_dim"]): parameters_to_optimize.append( sg.var(f'r{i}_p') )
            for i in range(opt_settings_dict["ghz_dim"]): parameters_to_optimize.append( sg.var(f'r{i}_i') )
            for i in range(opt_settings_dict["ghz_dim"]-2): parameters_to_optimize.append( sg.var(f'r{i}_by') )
            for i in range(opt_settings_dict["ghz_dim"]-3): parameters_to_optimize.append( sg.var(f'r{i}_bz') )
            
            self.post_rotations_range = range(num_of_non_rot_params , num_of_non_rot_params+ opt_settings_dict["ghz_dim"] )
            self.initial_rotations_range =  range(num_of_non_rot_params+ opt_settings_dict["ghz_dim"] 
                                                , num_of_non_rot_params+ 2*opt_settings_dict["ghz_dim"] )
            self.between_rotations_y_range = range(num_of_non_rot_params+ 2*opt_settings_dict["ghz_dim"] 
                                                , num_of_non_rot_params+ 3*opt_settings_dict["ghz_dim"] -2)
            self.between_rotations_z_range = range(num_of_non_rot_params+ 3*opt_settings_dict["ghz_dim"] -2 
                                                , num_of_non_rot_params+ 4*opt_settings_dict["ghz_dim"] -4)
        else:
            parameters_to_optimize.append( sg.var(f'r{0}_p') )
            parameters_to_optimize.append( sg.var(f'r{0}_i') )
            self.post_rotations_range = range(num_of_non_rot_params , num_of_non_rot_params+1 )
            self.initial_rotations_range =  range(num_of_non_rot_params+1,num_of_non_rot_params+2 )
            self.between_rotations_y_range = range(0)
            self.between_rotations_z_range = range(0)

        num_parameters = len(parameters_to_optimize)    
        
        global cost_function
        def cost_function(params):
            '''
            Function that is used for minimization. Essentially it only makes use of global gate performance function.
            '''
            # Obtain non rotational parameters
            tunable_parameters_dict = {}
            for i in range(num_of_non_rot_params):
                tunable_parameters_dict[parameters_to_optimize[i]] = params[i]
            
            # Obtain rotations values
            for i,ind in enumerate(self.post_rotations_range):
                tunable_parameters_dict[sg.var(f'r{i}_p')] = params[ind]
            

            for i,ind in enumerate(self.initial_rotations_range):
                tunable_parameters_dict[sg.var(f'r{i}_i')] = params[ind]


            for i,ind in enumerate(self.between_rotations_y_range):
                tunable_parameters_dict[sg.var(f'r{i}_by')] = params[ind]

            for i,ind in enumerate(self.between_rotations_z_range):
                tunable_parameters_dict[sg.var(f'r{i}_bz')] = params[ind]

            performance_dict = self.tunable_performance(tunable_parameters_dict,opt_settings_dict)
            
            cost = gate_simulation_functions.gate_performance_cost_function(performance_dict, opt_settings_dict)
            tunable_parameters_dict.clear()
            performance_dict.clear()
            return cost
        
        result = scipy.optimize.differential_evolution(cost_function,
                bounds=parameter_bounds,updating='deferred', workers=opt_settings_dict["n_cores"]
                ,disp=opt_settings_dict["disp_bool"],maxiter=opt_settings_dict["maxiter"])

        # Optimized parameters
        self.opt_tunable_dict = {parameters_to_optimize[i]: result.x[i] for i in range(num_parameters)}
            
        self.optimized_performance_dict = self.tunable_performance(self.opt_tunable_dict,opt_settings_dict)
        self.opt_tunable_dict[sg.var('DE')] = self.opt_tunable_dict[sg.var('DE_C')] * self.opt_tunable_dict[sg.var('C')]
        self.opt_tunable_dict[sg.var('tgs')] = self.optimized_performance_dict['gate_time']
        self.opt_tunable_dict[sg.var('rot')] = sg.real(self.rotation_hardware.subs(self.opt_tunable_dict))
        
        t_conf, memory_bool = gate_simulation_functions.time_interval_of_confidence(opt_settings_dict,self.optimized_performance_dict)
              
        self.optimized_performance_dict['t_conf'] = t_conf
        self.optimized_performance_dict['swap_into_memory'] = memory_bool

        output_concurrence = gate_simulation_functions.concurrence_from_evolution(self.hardware_evolution_unitary,self.opt_tunable_dict)
        
        
        self.optimized_performance_dict['concurrence'] = output_concurrence


        

    def GHZ_obtain_symbolic_fidelity_from_evolution(self,ghz_dims):
        '''
        Returns the fidelity of a GHZ state given a unitary when that is in symbolic form
        '''
        fidelity_dict_directory = 'saved_objects/expressions/ghz_fidelity_dict.pkl'
        
        if self.SimulationClass.load_analytical:
            with open(fidelity_dict_directory, 'rb') as inp:
                self.GHZ_fidelity_dict = pickle.load(inp)
        else:    
            self.GHZ_fidelity_dict = {}
            
            for ghz_dim in ghz_dims:
                print(f'Obtaining and saving expression for fidelity of GHZ-{ghz_dim}')
                n_qubits = ghz_dim
                GHZ_state = np.zeros(2**n_qubits)
                GHZ_state[[0,-1]] = 1/np.sqrt(2)
                GHZ_state_row_vec = sg.vector(sg.SR,GHZ_state).row()

                if self.symmetric:
                    evolution_symbols = [sg.var(f'u0'),sg.var(f'u1'),sg.var(f'u1'),sg.var(f'u2') ] 
                else:
                    evolution_symbols = [sg.var(f'u{i}') for i in range(self.GSRange) ] 
                evolution_symbolic = sg.Matrix(sg.SR,np.diag(evolution_symbols))

                rotation_symbolic = sg.var('rot')
                # Partially substitute tensoring functions for brevity
                ten_r_p = partial(gate_simulation_functions.ten_r ,n_qubits=n_qubits )
                ten_u_p = partial(gate_simulation_functions.ten_u ,n_qubits=n_qubits, evolution=evolution_symbolic )
                

                R_post = []
                if ghz_dim==2:
                    for i in range(ghz_dim): 
                        rot_val = (sg.var(f'r0_p') + rotation_symbolic )
                        R_post.append(gate_simulation_functions.R_z(rot_val))
                else:
                    for i in range(ghz_dim): 
                        rot_val = (sg.var(f'r{i}_p') + rotation_symbolic )
                        R_post.append(gate_simulation_functions.R_z(rot_val))
                
                R_init = []
                if ghz_dim==2:
                    for i in range(ghz_dim): 
                        rot_val = sg.var(f'r0_i') 
                        R_init.append(gate_simulation_functions.R_y(rot_val))
                else:
                    for i in range(ghz_dim): 
                        rot_val = sg.var(f'r{i}_i') 
                        R_init.append(gate_simulation_functions.R_y(rot_val))

                #between gates
                R_bet = []
                for i in range(ghz_dim-2): 
                    if i== ghz_dim-3:
                        #final only one ry needed
                        rot_val_y = sg.var(f'r{i}_by') 
                        Ry = ten_r_p( gate_simulation_functions.R_y(rot_val_y),0)
                        R_bet.append(Ry)
                    else:   
                        rot_val_y = sg.var(f'r{i}_by') 
                        Ry = ten_r_p( gate_simulation_functions.R_y(rot_val_y),0)
                        rot_val_z = sg.var(f'r{i}_bz') 
                        Rz = ten_r_p( gate_simulation_functions.R_z(rot_val_z),0)
                        R_bet.append(Ry*Rz)

                    

                H = sg.Matrix( qt.qip.operations.hadamard_transform(1).data.toarray() )
                
                plus_state = qt.Qobj(np.array([1,1])/np.sqrt(2) )
                init_state = qt.tensor(*(plus_state for i in range(n_qubits)))

                # Start at ++
                current_state = sg.vector(init_state.data.toarray().reshape(2**n_qubits)).column()

                # Apply initial adjustments
                for i in range(ghz_dim):
                    current_state = ten_r_p(R_init[i],i )* current_state

                
                
                for i in range(1,n_qubits):
                    # Apply gate (0,i) and post gate rotations
                    current_state =    ten_u_p((0,i))* current_state
                    current_state =    ten_r_p(H,i) * ten_r_p(R_post[i],i)* current_state
                    if R_bet:
                        #Apply in between gates
                        between_rot = R_bet.pop(0)
                        current_state = between_rot*current_state
                current_state = ten_r_p(R_post[0],0)* current_state
                
                aux_dict = {} # auxiliary dictionary for faster substitution
                aux_dict[sg.var('f0')] = sg.deepcopy(MMA_simplify(current_state[0,0]))
                aux_dict[sg.var('f1')] = sg.deepcopy(MMA_simplify(current_state[-1,0]))
                
                current_state[0,0]  = sg.var('f0',domain="complex")
                current_state[-1,0] = sg.var('f1',domain="complex")

                # normalize state
                current_state = current_state/  sg.vector(current_state).norm()

                fidelity_sg =  MMA_simplify(sg.abs_symbolic( (GHZ_state_row_vec * current_state)[0][0]  )  )
                
                self.GHZ_fidelity_dict[ghz_dim] =  {'fidelity':fidelity_sg , 'aux_dict' : aux_dict} 
            
            with open(fidelity_dict_directory, 'wb') as outp:
                pickle.dump(self.GHZ_fidelity_dict, outp, pickle.HIGHEST_PROTOCOL)