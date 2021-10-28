from operator import pos
from system import *

from qutip.qip.operations import rz

class Simulation():

    def __init__(self,setup_char):
        self.setup = system(setup_char,MMA=True,ManyVariables=False,TwoPhotonResonance= True)
        self.Variables_Declaration()
        self.Obtain_Effective_Hamiltonian()
        self.Obtain_Lindblads()
        self.ProjectOnGsDecSubspace()
        self.Create_Parameter_Dict()
        self.Numerical_Substitution()
        
        
    def Create_Parameter_Dict(self):  
        self.parameters = dict()
        self.variables = ['v','g_f','gamma','gamma_g','gamma_f','DE','De','phi','Omega','kappa_c','kappa_b']

        gamma_val = 1
        gamma_g_val = 0.1*gamma_val
        gamma_f_val = 0.9*gamma_val
        kappa_b_val = 100*gamma_val
        kappa_c_val = 100*gamma_val

        C_val = g**2/(kappa_c_val*gamma_val)
        
        Omega_val = sg.sqrt(C_val)*gamma_val*0.25 /2

        DE_val =  gamma_val/2*sg.sqrt(gamma_f_val) * sg.sqrt(4*C_val +gamma_f_val )  #gamma_val*sg.sqrt(C_val) # #  
        De_val =  C_val * gamma_val**2 /(2*DE_val) #gamma_val*sg.sqrt(C_val)  #  #
        phi_val = 0

        g_f_val = g
        v_val =   g*sg.sqrt(kappa_b_val*kappa_c_val)   # c=C :        g*sg.sqrt(kappa_b_val/gamma_val)

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
        self.eff_hamiltonian =  self.setup.eff_hamiltonian


    def Obtain_Lindblads(self):
        self.lind_op_number = len(self.setup.lindblad_list)
        self.eff_lind = []
        self.eff_lind_master_eq = []
        self.eff_lind_coeff = []
        print('Getting Lindblad Operators')
        for lind_op in range(self.lind_op_number ):
            self.eff_lind_coeff.append(self.setup.L_coeffs[lind_op])
            L_meq =  self.setup.eff_lindblad_list[lind_op]
            
            self.eff_lind_master_eq.append(L_meq)
            

    def ProjectOnGsDecSubspace(self):
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
        self.eff_hamiltonian = sg.Matrix(sg.SR,  self.eff_hamiltonian[ [k for k in GsDecPos] ,  [k for k in GsDecPos]] )

        for lind_op in range(self.lind_op_number ):  
            self.eff_lind_master_eq[lind_op] = sg.Matrix(sg.SR,  self.eff_lind_master_eq[lind_op][ [k for k in GsDecPos] ,  [k for k in GsDecPos]] )               
            


    def Run_Simulations(self,start,end,n, maximize_fidelity = True,equate_losses= False ):
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
            
            eff_hamiltonian_num = symround( self.eff_hamiltonian_C.subs(g=g_val).expand() ,digits=14,show_del=False)
            eff_hamiltonian_num = eff_hamiltonian_num.numpy().astype(float)
            H_obj = qt.Qobj(eff_hamiltonian_num)

            self.H_obj = H_obj

            #Initialize superoperator with hamiltonian
            L = -1j*( qt.spre(H_obj) -qt.spost(H_obj))

            L_obj_list = []
            L_np_list = []
            for lind_op in range(len(self.eff_lind_master_eq_C)):
                if equate_losses == False:
                    L_nparray = self.eff_lind_master_eq_C[lind_op].subs(g=g_val ).numpy().astype(complex)
                    L_np_list.append(L_nparray)
                    L_obj_list.append(qt.Qobj(L_nparray))
                else:
                    L_nparray = self.eff_lind_master_eq_C[lind_op].subs(g=g_val ).numpy().astype(complex)
                    x,y = np.nonzero(L_nparray)
                    for j in range(1,len(x)): #equate all elements to the 0th element
                        L_nparray[x[j],y[j]] = L_nparray[x[0],y[0]]

                    if str(self.eff_lind_coeff[lind_op]) != 'sqrt(gamma)': #neglect gamma losses  that are asymmetrical      
                        L_np_list.append(L_nparray)
                        L_obj_list.append(qt.Qobj(L_nparray))
                    else:
                        L_nparray *= 0 
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
            
            gate_time =  np.abs( np.pi  /(H[3]+H[0]-H[1]-H[2]) )
            

            Lt = (L*gate_time).expm()
            #post process rotations
            
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

            dm_f_gs =   1/ Psuccess * dm_f.eliminate_states(dec_pos) #ground state
            
            RotatedFinalState = R1.dag()*R2.dag()*psif_gs
            
            StateFidelity = qt.fidelity(dm_f_gs,RotatedFinalState)
                   

            f_prob = 1- Psuccess 
            infidelity = 1 - StateFidelity

            self.AVG_Infidelity[C_val_i] = infidelity
            self.AVG_P_failure[C_val_i] = f_prob
            self.AVG_gate_time[C_val_i] = gate_time

