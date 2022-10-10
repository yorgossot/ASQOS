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
        
        self.gate_time_symbolic =  sg.abs_symbolic( np.pi  /(H_symbolic[0]+H_symbolic[5]-H_symbolic[1]-H_symbolic[4]) ) * sg.var('tgr') #tgr : gate time ratio
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
        
        n_qubits = 3  # 2 qubit case
        plus_state = qt.Qobj(np.array([1,1])/np.sqrt(2) )
        init_state = qt.tensor(*(plus_state for i in range(n_qubits)))  

        # Start at +++
        init_state = sg.vector(init_state.data.toarray().reshape(2**n_qubits)).column()

        # Apply initial adjustments
        
        rot_val_0i = sg.var(f'r0_i') 
        R_matr = gate_simulation_functions.R_y(rot_val_0i)
        init_state = gate_simulation_functions.ten_r(R_matr,0,n_qubits )* init_state
        
        rot_val_1i = sg.var(f'r1_i') 
        R_matr = gate_simulation_functions.R_y(rot_val_1i)
        for q in [1,2]:
            init_state = gate_simulation_functions.ten_r(R_matr, q ,n_qubits )* init_state


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

        evolution_symbols = [sg.var(f'u00'),sg.var(f'u01'),sg.var(f'u01'),sg.var(f'u02'),sg.var('u10'),sg.var('u11'),sg.var('u11'),sg.var('u12') ] 
        evolution_symbolic = sg.Matrix(sg.SR,np.diag(evolution_symbols))
        self.evolution_symbolic = evolution_symbolic
        # Post process rotations
        self.rotation_symbolic_HL_0 =  gate_time_symbolic * (H_symbolic[0] - H_symbolic[4]) 
        self.rotation_symbolic_0 = sg.var('rot0')

        self.rotation_symbolic_HL_1 =  gate_time_symbolic * (H_symbolic[0] - H_symbolic[1]) 
        self.rotation_symbolic_1 = sg.var('rot1')
        # The following expressions are not fully expressed in terms of detunings necessarily.
        
        self.GHZ_obtain_symbolic_fidelity_from_evolution() 





    def obtain_gate_performance_hardware(self, hardware_dict):
        '''
        Obtain the gate performance given the hardware setting. Detunings are still allowed to vary.
        '''
        # A dictionary of the hardware setting
        self.hardware_dict = hardware_dict


        De01_val =  sg.var('De1') - hardware_dict[sg.var('D_max')]
        De02_val =  sg.var('De2') - hardware_dict[sg.var('D_max')]

        
        # Contains parametric De0, thus not added to self.hardware_dict
        hw_subs_dict = {**hardware_dict,sg.var('De01'): De01_val,sg.var('De02'): De02_val }
        

        self.substitution_dict_hw = {}
        for diag,var in enumerate(self.H_symbolic):
            self.substitution_dict_hw[var] = self.basic_substitution(self.setup.eff_hamiltonian_gs[diag,diag]).subs(hw_subs_dict) 
        
        n_qubits = 3 #direct GHZ
        for elem,var in enumerate(self.L_symbolic):
            lind_op = elem // (2**n_qubits)
            which   = elem % (2**n_qubits)
            L_num = self.EffLindbladElements[lind_op][which]
            if type(L_num) is type(sg.var('x')): L_num = L_num.subs(hw_subs_dict)
            self.substitution_dict_hw[var] = L_num
        
        # Obtain unitary in terms of hardware parameters
        self.hardware_evolution_unitary = self.evolution_symbolic_HL.subs(self.substitution_dict_hw)

        self.substitution_dict_hw[sg.var(f'u00')] = self.hardware_evolution_unitary[0,0]
        self.substitution_dict_hw[sg.var(f'u01')] = self.hardware_evolution_unitary[1,1]
        self.substitution_dict_hw[sg.var(f'u02')] = self.hardware_evolution_unitary[3,3]
        self.substitution_dict_hw[sg.var(f'u10')] = self.hardware_evolution_unitary[4,4]
        self.substitution_dict_hw[sg.var(f'u11')] = self.hardware_evolution_unitary[5,5]
        self.substitution_dict_hw[sg.var(f'u12')] = self.hardware_evolution_unitary[7,7]

        self.rotation_0_hardware = self.rotation_symbolic_HL_0(self.substitution_dict_hw)
        self.substitution_dict_hw[sg.var('rot0')] = self.rotation_0_hardware       

        self.rotation_1_hardware = self.rotation_symbolic_HL_1(self.substitution_dict_hw)
        self.substitution_dict_hw[sg.var('rot1')] = self.rotation_1_hardware




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

            aux_dict = self.GHZ_fidelity_dict['aux_dict'].copy()
            for key in aux_dict:
                HL_dict[key] = aux_dict[key].subs(HL_dict).subs(tgs=gate_time)
            
            for i in range(2):
                for j in range(3):
                    HL_dict[sg.var(f'u{i}{j}')] = HL_dict[sg.var(f'u{i}{j}')].subs(HL_dict)

            
            HL_dict[sg.var('rot0')] = self.rotation_0_hardware.subs(HL_dict)
            HL_dict[sg.var('rot1')] = self.rotation_0_hardware.subs(HL_dict)
            
            fid_expression = self.GHZ_fidelity_dict['fidelity']
               
            fidelity =   float(sg.real(fid_expression.subs(HL_dict).subs( tgs=gate_time)))

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
        
        parameters_to_optimize = [sg.var('C'),sg.var('De1'),sg.var('De2'),  sg.var('DE_C'), sg.var('tgr')]
        num_of_non_rot_params = len(parameters_to_optimize)
        # Add rotation parameters depending on the ghz_dim


        for i in range(2):
            parameters_to_optimize.append( sg.var(f'r{i}_p') )
            parameters_to_optimize.append( sg.var(f'r{i}_i') )
        self.post_rotations_range = range(num_of_non_rot_params , num_of_non_rot_params+2 )
        self.initial_rotations_range =  range(num_of_non_rot_params+2,num_of_non_rot_params+4 )


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
        self.opt_tunable_dict[sg.var('rot0')] = sg.real(self.rotation_0_hardware.subs(self.opt_tunable_dict))
        self.opt_tunable_dict[sg.var('rot1')] = sg.real(self.rotation_0_hardware.subs(self.opt_tunable_dict))
        
        t_conf, _ = gate_simulation_functions.time_interval_of_confidence(opt_settings_dict,self.optimized_performance_dict)
              
        self.optimized_performance_dict['t_conf'] = t_conf
        
        


        
 
    def GHZ_obtain_symbolic_fidelity_from_evolution(self):
        '''
        Returns the fidelity of a GHZ state given a unitary when that is in symbolic form
        '''
        direct_ghz_fidelity_directory = 'saved_objects/expressions/direct_ghz_fidelity.pkl'
        
        if self.SimulationClass.load_analytical:
            with open(direct_ghz_fidelity_directory, 'rb') as inp:
                self.GHZ_fidelity_dict = pickle.load(inp)
        else:    
            ghz_dim = 3
            print(f'Obtaining and saving expression for fidelity of direct GHZ-{ghz_dim}')
            n_qubits = ghz_dim
            GHZ_state = np.zeros(2**n_qubits)
            GHZ_state[[0,-1]] = 1/np.sqrt(2)
            GHZ_state_row_vec = sg.vector(sg.SR,GHZ_state).row()


            evolution_symbols = [sg.var(f'u00'),sg.var(f'u01'),sg.var(f'u01'),sg.var(f'u02'),sg.var('u10'),sg.var('u11'),sg.var('u11'),sg.var('u12') ] 
           
            evolution_symbolic = sg.Matrix(sg.SR,np.diag(evolution_symbols))

            rotation0_symbolic = sg.var('rot0')
            rotation1_symbolic = sg.var('rot1')
            # Partially substitute tensoring functions for brevity
            ten_r_p = partial(gate_simulation_functions.ten_r ,n_qubits=n_qubits )
            

            R_post = []
            rot0_val_i = (sg.var(f'r0_p') + rotation0_symbolic )
            R_post.append(gate_simulation_functions.R_z(rot0_val_i))
            for q in [1,2]: 
                rot1_val_i = (sg.var(f'r1_p') + rotation1_symbolic )
                R_post.append(gate_simulation_functions.R_z(rot1_val_i))
            
            R_init = []
            rot0_val_p = sg.var(f'r0_i') 
            R_init.append(gate_simulation_functions.R_y(rot0_val_p))
            for q in [1,2]: 
                rot1_val_p = sg.var(f'r1_i') 
                R_init.append(gate_simulation_functions.R_y(rot1_val_p))
                

            H = sg.Matrix( qt.qip.operations.hadamard_transform(1).data.toarray() )
            
            plus_state = qt.Qobj(np.array([1,1])/np.sqrt(2) )
            init_state = qt.tensor(*(plus_state for i in range(n_qubits)))

            # Start at ++
            current_state = sg.vector(init_state.data.toarray().reshape(2**n_qubits)).column()

            # Apply initial adjustments
            for i in range(ghz_dim):
                current_state = ten_r_p(R_init[i],i )* current_state

            current_state = evolution_symbolic*current_state
            
            for i in range(1,n_qubits):
                current_state =    ten_r_p(H,i) * ten_r_p(R_post[i],i)* current_state
                
            current_state = ten_r_p(R_post[0],0)* current_state
            
            aux_dict = {} # auxiliary dictionary for faster substitution
            aux_dict[sg.var('f0')] = sg.deepcopy(MMA_simplify(current_state[0,0]))
            aux_dict[sg.var('f1')] = sg.deepcopy(MMA_simplify(current_state[-1,0]))
            
            current_state[0,0]  = sg.var('f0',domain="complex")
            current_state[-1,0] = sg.var('f1',domain="complex")

            # normalize state
            current_state = current_state/  sg.vector(current_state).norm()

            fidelity_sg =  MMA_simplify(sg.abs_symbolic( (GHZ_state_row_vec * current_state)[0][0]  )  )
            
            self.GHZ_fidelity_dict =  {'fidelity':fidelity_sg , 'aux_dict' : aux_dict} 
        
        with open(direct_ghz_fidelity_directory, 'wb') as outp:
            pickle.dump(self.GHZ_fidelity_dict, outp, pickle.HIGHEST_PROTOCOL)