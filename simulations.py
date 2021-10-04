from system import *

class Simulation():

    def __init__(self,setup_char):
        self.setup = system(setup_char,MMA=True,ManyVariables=False,TwoPhotonResonance= True)

        self.Variables_Declaration()
        self.Obtain_Effective_Hamiltonian()
        self.Obtain_Lindblads()
        self.Numerical_Substitution()
        self.Run_Simulations()


    def Numerical_Substitution(self):
        def substitute(a):
            variables = ['c','R_f','gamma','r_g','r_f','kappa_c','kappa_b','DEg','Deg','phi','Omega']
            gamma_val = 1
            c_val = C
            Omega_val = sg.sqrt(C)*gamma_val*0.1
            R_f_val = 1
            r_g_val = 1
            r_f_val = 1
            kappa_c_val = 200*gamma_val
            kappa_b_val = 200*gamma_val
            DEg_val = sg.sqrt(C)/gamma_val
            Deg_val = sg.sqrt(C)/gamma_val
            phi_val = 0
            q = sg.copy(a)
            for var in variables:
                q = eval(f'q.subs({var}={var}_val )') 
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
        sg.var('gamma','DE','De','g','g_f','Omega','v','gamma_f','gamma_g','gamma0','De0','g0',domain='positive')
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

        for diag in range(4):

            b = self.setup.eff_hamiltonian_gs[diag,diag]


            b = b.operands()[1].operands()[0]  


            b = b.subs(DE=DEg*gamma)
            b = b.subs(De=Deg*gamma)
   
            b = b.subs(g_f=g*sg.sqrt(R_f))
            b = b.subs(g0=g*sg.sqrt(R0))
            b = b.subs(v=g*sg.sqrt(R_v))
            b = b.subs(kappa_b=gamma/r_b)  
            b = b.subs(gamma_g=gamma*r_g)
            b = b.subs(gamma_f=gamma*r_f)
            b = b.subs(gamma0=gamma*r0)
            b = b.subs(g = sg.sqrt(C*kappa_c*gamma)) 
            b = b.subs(r_b = c/(C*R_v))
            b = b*gamma  #Omega has to be substituted with Omega/gamma


            b = b._mathematica_().Factor()._sage_()
            b = MMA_simplify(b,full= False)
            
            b = symround(b,digits=14,show_del=False)
            
            b = (b + sg.conjugate(b) ) * (-0.5) * Omega**2
            
            b = b._mathematica_().Together()._sage_()
            
            b = symround(MMA_simplify(b,full= False),digits=14)
            
            num = b._mathematica_().Numerator()._sage_()
            num = symround( MMA_simplify(num.expand(),full= False) ,digits=14,show_del=False)
            
            den = b._mathematica_().Denominator()._sage_()
            den = symround(MMA_simplify(den.expand(),full= False),digits=14,show_del=False)
            
            b = num/den
            
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
                L_elem = L_elem.subs(De=Deg*gamma)
                L_elem = L_elem.subs(DE=DEg*gamma)
                #COUPLINGS
                L_elem = L_elem.subs(g_f=g*sg.sqrt(R_f))
                L_elem = L_elem.subs(v=g*sg.sqrt(R_v))
                #ABSORBTIONS
                L_elem = L_elem.subs(kappa_b=gamma/r_b)
                L_elem = L_elem.subs(gamma_g=gamma*r_g)
                L_elem = L_elem.subs(gamma_f=gamma*r_f)
                L_elem = L_elem.subs(g = sg.sqrt(C*gamma*kappa_c)) 
                L_elem = L_elem.subs(r_b = c/(C*R_v))               
                L_elem = L_elem._mathematica_().Factor()._sage_()

                L_elem = symround(MMA_simplify(L_elem),digits=14,show_del=False)
                L_elem = sg.SR(str(L_elem).replace('Sqrt','sqrt'))               
                L_elem = symround(L_elem,digits=14 ,show_del=False)
                
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


    def Run_Simulations(self):
        self.C_val_range = np.logspace(0,3,num=40)
        self.AVG_P_failure = np.zeros(np.shape(self.C_val_range))
        for (C_val_i,C_val) in enumerate(self.C_val_range):      
            print(f'Simulation for C={C_val}') 
            eff_hamiltonian_num = symround( self.eff_hamiltonian_C.subs(C=C_val) ,digits=14,show_del=False)
            eff_hamiltonian_num = eff_hamiltonian_num.numpy().astype(float)
            H_obj = qt.Qobj(eff_hamiltonian_num)
            

            L_obj_list = []
            L_np_list = []
            for lind_op in range(self.lind_op_number):
                L_nparray = self.eff_lind_master_eq_C[lind_op].subs(C=C_val ).numpy().astype(complex)
                L_np_list.append(L_nparray)
                L_obj_list.append(qt.Qobj(L_nparray))
            
            qubit_cardinal_states = [np.array([1,0]),np.array([0,1])\
                ,np.array([1,1])/np.sqrt(2),np.array([1,-1])/np.sqrt(2)\
                ,np.array([1,1j])/np.sqrt(2),np.array([1,-1j])/np.sqrt(2) ]
            
            #two_qubit_cardinal_states = np.kron(qubit_cardinal_states,qubit_cardinal_states)
            pp_state = np.kron(qubit_cardinal_states[2],qubit_cardinal_states[2])

            f_prob = 0
            for state  in [pp_state]:  #in two_qubit_cardinal_states : #also divide by len
                dim5_vec = np.zeros(5,dtype=complex)
                dim5_vec[0:4] = state
                psi0 = qt.Qobj(dim5_vec)
                gate_time =  np.abs( np.pi\
                    /(eff_hamiltonian_num[3,3]+eff_hamiltonian_num[0,0]-eff_hamiltonian_num[1,1]-eff_hamiltonian_num[2,2]))

                times = np.linspace(0,gate_time,100)

                F_proj  = qt.Qobj(np.diag(np.array([0,0,0,0,1])))

                sol = qt.mesolve(H_obj, psi0, times,L_obj_list,  [F_proj])
                f_prob += sol.expect[0][-1]
            
            f_prob = f_prob #/ len(pp)

            self.AVG_P_failure[C_val_i] = f_prob
