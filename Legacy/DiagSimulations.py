#
# File containing a class to do simulations using \gamma_g=0 .
#


from system import *
from qutip.qip.operations import rz

class Simulation():

    def __init__(self,setup_char):
        self.setup_char = setup_char
        self.setup = system(setup_char,MMA=True,ManyVariables=False,TwoPhotonResonance= True)
        self.Variables_Declaration()
        self.Create_Parameter_Dict()
        self.Obtain_Lindblads()
        self.ObtainGatePerformanceAnalytically()
        
        
    def Create_Parameter_Dict(self):  
        self.parameters = dict()
        self.variables = ['v','g','g_f','gamma','gamma_g','gamma_f','phi','Omega','kappa_c','kappa_b','g0','gamma0']

        gamma_val = 1
        gamma0_val = gamma_val
        gamma_g_val = 0 # DO NOT CHANGE
        gamma_f_val = 1 * gamma_val
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


    def BasicSubstitution(self, Param = None):   
        
        def substitute(a):
            q = sg.copy(a)
            for var in self.variables:
                q = eval(f"q.subs({var}= self.parameters['{var}'] )") 
            return q

        if Param is not None:
            return substitute(Param)


    def Variables_Declaration(self):
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


    def Obtain_Lindblads(self):
        self.lind_op_number = len(self.setup.lindblad_list)
        self.eff_lind = []
        self.EffLindbladElements = []
        self.eff_lind_coeff = []
        print('Simplifying Lindblad Operators')
        for lind_op in range(self.lind_op_number ):
            print(f'{lind_op} out of {self.lind_op_number-1}')
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
                L_meq[state] += self.BasicSubstitution(L_nonzeros[i])
             
                    
            self.EffLindbladElements.append(L_meq)       
            
          


    def ObtainGatePerformanceAnalytically(self):

        GSRange = range(self.setup.gs_dim)

        H = [self.BasicSubstitution(self.setup.eff_hamiltonian_gs[diag,diag]) for diag in GSRange]


        gate_time =  sg.abs_symbolic( np.pi  /(H[3]+H[0]-H[1]-H[2]) ) * sg.var('tgr') #tgr : gate time ratio
        
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

        self.StateFidelity  = sg.abs_symbolic(RotatedFinalVector*FinalVector)
        self.PSuccess       = PSuccess
        self.GateTime       = gate_time



    def ObtainGatePerformanceAnalyticallyC(self,C_val , c_val):

        GSRange = range(self.setup.gs_dim)

        H = [self.BasicSubstitution(self.setup.eff_hamiltonian_gs[diag,diag]).subs(C=C_val,c=c_val) for diag in GSRange]


        gate_time =  sg.abs_symbolic( np.pi  /(H[3]+H[0]-H[1]-H[2]) ) * sg.var('tgr') #tgr : gate time ratio
        
        gate_time_symbolic = sg.var('tgs')

        EffEffHamiltonian = [-1j*H[i] for i in GSRange ]

        LossFactors = [0 for i in GSRange]
        

        for lind_op in range(self.lind_op_number):
            for which in GSRange:
                
                L_num = self.EffLindbladElements[lind_op][which]
                if type(L_num) is type(sg.var('x')): L_num = L_num.subs(C=C_val,c=c_val)
                
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

        self.StateFidelityC  = sg.abs_symbolic(RotatedFinalVector*FinalVector)
        self.PSuccessC       = PSuccess
        self.GateTimeC       = gate_time