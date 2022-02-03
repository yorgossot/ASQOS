from system import *

from qutip.qip.operations import rz

class Simulation():

    def __init__(self,setup_char):
        self.setup = system(setup_char,MMA=True,ManyVariables=False,TwoPhotonResonance= True)
        self.Variables_Declaration()
        self.Obtain_Effective_Hamiltonian()
        self.Obtain_Lindblads()
        self.Create_Parameter_Dict()
        self.Numerical_Substitution()
        
    def Create_Parameter_Dict(self):  
        self.parameters = dict()
        self.variables = ['v','g_f','gamma','gamma_g','gamma_f','DE','De','phi','Omega','kappa_c','kappa_b','De0','g0','gamma0']

        gamma_val = 1
        gamma0_val = gamma_val
        gamma_g_val = 0 # DO NOT CHANGE
        gamma_f_val = 1*gamma_val
        kappa_b_val = 100*gamma_val
        kappa_c_val = 100*gamma_val
        
        g = sg.var('g') 
        C_val = g**2/(kappa_c_val*gamma_val)
        
        Omega_val = sg.sqrt(C_val)*gamma_val*0.25 /2

        DE_val  = sg.sqrt(4*C_val +gamma_f_val )/2 # gamma_val/2*sg.sqrt(gamma_f_val) #gamma_val*sg.sqrt(C_val) #* sg.sqrt(4*C_val +gamma_f_val )  #gamma_val*sg.sqrt(C_val) # #  
        De_val  = C_val * gamma_val**2 /(2*DE_val)#gamma_val*sg.sqrt(C_val) # #gamma_val*sg.sqrt(C_val)#10*gamma_val#C_val * gamma_val**2 /(2*DE_val) #gamma_val*sg.sqrt(C_val)  #  #
        De0_val = -gamma_val*sg.sqrt(C_val) #De_val + 10*gamma#-(gamma_val*sg.sqrt(C_val) )#+ gamma_val*10)
        phi_val = 0

        g_f_val = g
        g0_val  = g
        v_val   = g*sg.sqrt(kappa_b_val*kappa_c_val)   # c=C :        g*sg.sqrt(kappa_b_val/gamma_val)

        for var in self.variables:
            exec(f"self.parameters['{var}'] = {var}_val ")


    def Numerical_Substitution(self):   
        
        def substitute(a):
            q = sg.copy(a)
            for var in self.variables:
                q = eval(f"q.subs({var}= self.parameters['{var}'] )") 
            return q
            
        self.eff_hamiltonian_C = symround( substitute(self.eff_hamiltonian) , digits=14,show_del=False)

        self.eff_lind_master_eq_C = []
        for lind_op in range(self.lind_op_number):
            self.eff_lind_master_eq_C.append(  substitute(self.eff_lind_master_eq[lind_op])     )

    def Variables_Declaration(self):
        sg.var('DEg',domain='positive',  latex_name =r'\Delta_{E\gamma}')
        sg.var('Deg',domain='positive',  latex_name =r'\Delta_{e\gamma}')
        sg.var('c',domain='positive',  latex_name =r'c')
        sg.var('C',domain='positive',  latex_name =r'C')
        sg.var('gamma','DE','De','g','g_f','Omega','v','gamma_f','gamma_g','gamma0','De0','phi','g0',domain='positive')
        sg.var('r0',domain='real',latex_name =r'r_0')
        sg.var('R_f',domain='real')#ratio  (g_f/g)^2
        sg.var('R0',domain='positive',  latex_name =r'R_0')
        sg.var('R_v',domain='real',  latex_name =r'R_{\nu}') #ratio (v/g)^2
        sg.var('r_g',domain='real',latex_name =r'r_g')
        sg.var('r_b',domain='real')
        sg.var('r_f',domain='real',latex_name =r'r_f')
        sg.var('kappa_c','kappa_b',domain='positive')


    def Obtain_Effective_Hamiltonian(self):
        self.eff_hamiltonian =  sg.matrix(sg.SR, self.setup.gs_dim+1,self.setup.gs_dim+1) 

        for diag in range(self.setup.gs_dim):

            b = self.setup.eff_hamiltonian_gs[diag,diag]
            
            self.eff_hamiltonian[diag,diag] =  b

    def Obtain_Lindblads(self):
        self.lind_op_number = len(self.setup.lindblad_list)
        self.eff_lind = []
        self.eff_lind_master_eq = []
        self.eff_lind_coeff = []
        print('Simplifying Lindblad Operators')
        for lind_op in range(self.lind_op_number ):
            print(f'{lind_op} out of {self.lind_op_number-1}')
            self.eff_lind.append([])
            L_matrix = self.setup.eff_lindblad_list[lind_op]
            L_nonzeros = []
            L_nonzeros_pos = []
            for i in  range(L_matrix.nrows()):
                for j in  range(L_matrix.ncols()):
                    if not L_matrix[i,j].is_zero():
                        L_nonzeros.append(L_matrix[i,j])
                        L_nonzeros_pos.append((i,j))
            
            self.eff_lind_coeff.append(self.setup.L_coeffs[lind_op])
            
            gs_dim = self.setup.gs_dim
            gs_pos_correspondence = self.setup.pos_gs
            L_meq = sg.matrix(sg.SR, gs_dim+1,gs_dim+1) #extra dimension for heralded errors
            
                
            for which in range(len(L_nonzeros)):
                L_elem = L_nonzeros[which]
                
                self.eff_lind[lind_op].append(L_elem)
                
                #separate gamma_g errors from the rest so that we can project the rest of them to a 
                #one-dimensional decayed subspace
                if L_nonzeros_pos[which][0] not in self.setup.pos_gs:
                    index_i = gs_dim 
                    index_j = gs_pos_correspondence.index(L_nonzeros_pos[which][1])
                    L_meq[index_i,index_j] = L_elem
                else:
                    index_i = gs_pos_correspondence.index(L_nonzeros_pos[which][0])
                    index_j = gs_pos_correspondence.index(L_nonzeros_pos[which][1])
                    L_meq[index_i,index_j] = L_elem
                
                    
            self.eff_lind_master_eq.append(L_meq)        


    '''def Obtain_Variables(self):
        self.parameter_list =  list( str(elem) for elem in list(self.eff_hamiltonian.variables()))
        for leff in self.eff_lind_master_eq:
            params = list( str(elem) for elem in list(leff.variables()))
            for var in params:
                if var not in self.parameter_list:
                    self.parameter_list.append(var)
        self.parameters = dict.fromkeys(self.parameter_list)
'''

    def Run_Simulations(self,start,end,n, maximize_fidelity = True ):
        '''
        Run simulations for a specified logarithmic range of cooperativities.
        '''
        self.C_val_range = np.logspace(start,end,num=n)
        self.AVG_P_failure = np.zeros(np.shape(self.C_val_range))
        self.AVG_Infidelity = np.zeros(np.shape(self.C_val_range))
        self.AVG_gate_time = np.zeros(np.shape(self.C_val_range))
        print(f'Starting qutip simulations for C in {self.C_val_range[0]}-{self.C_val_range[-1]}') 
        for (C_val_i,C_val) in enumerate(self.C_val_range):      
            
            g_val = sg.sqrt(C_val*self.parameters['kappa_c'] *self.parameters['gamma'] )
            
            eff_hamiltonian_num = symround( 2*self.eff_hamiltonian_C.subs(g=g_val).expand() ,digits=14,show_del=False)
            eff_hamiltonian_num = eff_hamiltonian_num.numpy().astype(float)
            H_obj = qt.Qobj(eff_hamiltonian_num)
            

            L_obj_list = []
            L_np_list = []
            for lind_op in range(len(self.eff_lind_master_eq_C)):
                
                L_nparray = self.eff_lind_master_eq_C[lind_op].subs(g=g_val ).numpy().astype(complex)
                L_np_list.append(L_nparray)
                L_obj_list.append(qt.Qobj(L_nparray))
                '''
                
                if lind_op ==2:
                    L_nparray = self.eff_lind_master_eq_C[lind_op].subs(g=g_val ).numpy().astype(complex)
                    L_nparray[1,1] = L_nparray[0,0]
                    L_nparray[2,2] = L_nparray[0,0]
                    L_nparray[3,3] = L_nparray[0,0]
                    #L_nparray[4,4] = L_nparray[0,0]
                    L_np_list.append(L_nparray)
                    L_obj_list.append(qt.Qobj(L_nparray))
                else:

                    L_nparray = self.eff_lind_master_eq_C[lind_op].subs(g=g_val ).numpy().astype(complex)
                    L_nparray[4,1] = L_nparray[4,3]
                    L_nparray[4,2] = L_nparray[4,3]
                    L_nparray[4,0] = L_nparray[4,3]
                    #L_nparray[4,4] = L_nparray[4,0]
                    L_np_list.append(L_nparray)
                    L_obj_list.append(qt.Qobj(L_nparray))             
                '''         
                
            self.L_obj_list = L_obj_list
                
            
            qubit_cardinal_states = [np.array([1,0]),np.array([0,1])\
                ,np.array([1,1])/np.sqrt(2),np.array([1,-1])/np.sqrt(2)\
                ,np.array([1,1j])/np.sqrt(2),np.array([1,-1j])/np.sqrt(2) ]
            
            #two_qubit_cardinal_states = np.kron(qubit_cardinal_states,qubit_cardinal_states)
            pp_state = np.array([1,1,1,1])/2

            psif_gs  =  qt.Qobj(np.array([1,1,1,-1])/2)  #target state
            
            Psuccess = 0
            StateFidelity = 0
  
            dim5_vec = np.zeros(5,dtype=complex)
            dim5_vec[0:4] = pp_state
            psi0 = qt.Qobj(dim5_vec)
            gate_time =  np.abs( np.pi\
                /(eff_hamiltonian_num[3,3]+eff_hamiltonian_num[0,0]-eff_hamiltonian_num[1,1]-eff_hamiltonian_num[2,2]))
            #post process rotations
            r1 = -gate_time * (eff_hamiltonian_num[0,0] - eff_hamiltonian_num[2,2])
            R1 = qt.tensor( rz(r1 ),qt.identity(2)) 
            R1_5 = np.zeros((5,5),dtype='complex128')
            R1_5[0:4,0:4] = R1.full()
            R1_5[4,4] = 1
            R1 = qt.Qobj(R1_5)
            
            r2 = -gate_time * (eff_hamiltonian_num[0,0] - eff_hamiltonian_num[1,1])
            R2 = qt.tensor(qt.identity(2),rz(r2 )) 
            R2_5 = np.zeros((5,5),dtype='complex128')
            R2_5[0:4,0:4] = R2.full()
            R2_5[4,4] = 1
            R2 = qt.Qobj(R2_5)
            #simulate
            
            try:
                if maximize_fidelity == False:
                    times = np.linspace(0,gate_time,3)
                    sol = qt.mesolve(H_obj, psi0, times,L_obj_list, [] )
                    #final density matrix 
                    dm_f = sol.states[-1]
                    dm_ff =  R2*R1*dm_f*R1.dag()*R2.dag() # add rotations

                    Psuccess = 1- dm_ff[4,4]

                    dm_f_gs =   1/ Psuccess * dm_ff.eliminate_states([4]) #ground state

                    StateFidelity = qt.fidelity(dm_f_gs,psif_gs)
                else:                       
                    #extra_time_points = np.linspace(0.98,1.08,20)*gate_time
                    times = np.linspace(0,gate_time*1.1,111)
                    sol = qt.mesolve(H_obj, psi0, times,L_obj_list, [] )
                    f = np.zeros(np.shape(times))
                    p = np.zeros(np.shape(times))                       
                    for j in range(len(times)):
                        dm_f = sol.states[j]
                        dm_ff =  R2*R1*dm_f*R1.dag()*R2.dag() # add rotations                           
                        Ps = 1- dm_ff[4,4]
                        dm_f_gs =   1/ Ps * dm_ff.eliminate_states([4]) #ground state
                        StateFid = qt.fidelity(dm_f_gs,psif_gs)
                        f[j] = StateFid
                        p[j] = Ps
                    t = np.argmax(f)
                    StateFidelity = f[t]
                    Psuccess = p[t]
                    gate_time = times[t]

            except:
                print(f'Error in qutip for C={C_val}')
                    

            f_prob = 1- Psuccess 
            infidelity = 1 - StateFidelity

            self.AVG_Infidelity[C_val_i] = infidelity
            self.AVG_P_failure[C_val_i] = f_prob
            self.AVG_gate_time[C_val_i] = gate_time

    def Simulate(self,C_val, maximize_fidelity = True ):
        '''
        Run simulations for a specified logarithmic range of cooperativities.
        '''      
            
        g_val = sg.sqrt(C_val*self.parameters['kappa_c'] *self.parameters['gamma'] )
        
        eff_hamiltonian_num = symround( 2*self.eff_hamiltonian_C.subs(g=g_val).expand() ,digits=14,show_del=False)
        eff_hamiltonian_num = eff_hamiltonian_num.numpy().astype(float)
        H_obj = qt.Qobj(eff_hamiltonian_num)
        

        L_obj_list = []
        L_np_list = []
        for lind_op in range(len(self.eff_lind_master_eq_C)):
            
            L_nparray = self.eff_lind_master_eq_C[lind_op].subs(g=g_val ).numpy().astype(complex)
            L_np_list.append(L_nparray)
            L_obj_list.append(qt.Qobj(L_nparray))
            '''
            
            if lind_op ==2:
                L_nparray = self.eff_lind_master_eq_C[lind_op].subs(g=g_val ).numpy().astype(complex)
                L_nparray[1,1] = L_nparray[0,0]
                L_nparray[2,2] = L_nparray[0,0]
                L_nparray[3,3] = L_nparray[0,0]
                #L_nparray[4,4] = L_nparray[0,0]
                L_np_list.append(L_nparray)
                L_obj_list.append(qt.Qobj(L_nparray))
            else:

                L_nparray = self.eff_lind_master_eq_C[lind_op].subs(g=g_val ).numpy().astype(complex)
                L_nparray[4,1] = L_nparray[4,3]
                L_nparray[4,2] = L_nparray[4,3]
                L_nparray[4,0] = L_nparray[4,3]
                #L_nparray[4,4] = L_nparray[4,0]
                L_np_list.append(L_nparray)
                L_obj_list.append(qt.Qobj(L_nparray))             
            '''         
            
        self.L_obj_list = L_obj_list
            
        
        qubit_cardinal_states = [np.array([1,0]),np.array([0,1])\
            ,np.array([1,1])/np.sqrt(2),np.array([1,-1])/np.sqrt(2)\
            ,np.array([1,1j])/np.sqrt(2),np.array([1,-1j])/np.sqrt(2) ]
        
        #two_qubit_cardinal_states = np.kron(qubit_cardinal_states,qubit_cardinal_states)
        pp_state = np.array([1,1,1,1])/2

        psif_gs  =  qt.Qobj(np.array([1,1,1,-1])/2)  #target state
        
        Psuccess = 0
        StateFidelity = 0

        dim5_vec = np.zeros(5,dtype=complex)
        dim5_vec[0:4] = pp_state
        psi0 = qt.Qobj(dim5_vec)
        gate_time =  np.abs( np.pi\
            /(eff_hamiltonian_num[3,3]+eff_hamiltonian_num[0,0]-eff_hamiltonian_num[1,1]-eff_hamiltonian_num[2,2]))
        #post process rotations
        r1 = -gate_time * (eff_hamiltonian_num[0,0] - eff_hamiltonian_num[2,2])
        R1 = qt.tensor( rz(r1 ),qt.identity(2)) 
        R1_5 = np.zeros((5,5),dtype='complex128')
        R1_5[0:4,0:4] = R1.full()
        R1_5[4,4] = 1
        R1 = qt.Qobj(R1_5)
        
        r2 = -gate_time * (eff_hamiltonian_num[0,0] - eff_hamiltonian_num[1,1])
        R2 = qt.tensor(qt.identity(2),rz(r2 )) 
        R2_5 = np.zeros((5,5),dtype='complex128')
        R2_5[0:4,0:4] = R2.full()
        R2_5[4,4] = 1
        R2 = qt.Qobj(R2_5)
        #simulate
        
        try:
            if maximize_fidelity == False:
                times = np.linspace(0,gate_time,3)
                sol = qt.mesolve(H_obj, psi0, times,L_obj_list, [] )
                #final density matrix 
                dm_f = sol.states[-1]
                dm_ff =  R2*R1*dm_f*R1.dag()*R2.dag() # add rotations

                Psuccess = 1- dm_ff[4,4]

                dm_f_gs =   1/ Psuccess * dm_ff.eliminate_states([4]) #ground state

                StateFidelity = qt.fidelity(dm_f_gs,psif_gs)
            else:                       
                #extra_time_points = np.linspace(0.98,1.08,20)*gate_time
                times = np.linspace(0,gate_time*1.1,111)
                sol = qt.mesolve(H_obj, psi0, times,L_obj_list, [] )
                f = np.zeros(np.shape(times))
                p = np.zeros(np.shape(times))                       
                for j in range(len(times)):
                    dm_f = sol.states[j]
                    dm_ff =  R2*R1*dm_f*R1.dag()*R2.dag() # add rotations                           
                    Ps = 1- dm_ff[4,4]
                    dm_f_gs =   1/ Ps * dm_ff.eliminate_states([4]) #ground state
                    StateFid = qt.fidelity(dm_f_gs,psif_gs)
                    f[j] = StateFid
                    p[j] = Ps
                t = np.argmax(f)
                StateFidelity = f[t]
                Psuccess = p[t]
                gate_time = times[t]

        except:
            print(f'Error in qutip for C={C_val}')
                

        f_prob = 1- Psuccess 
        infidelity = 1 - StateFidelity

        return gate_time , f_prob, infidelity
