#
# File containing all necessary classes to simulate the gate through jupyter notebooks.
#

from ...system import system
from . import ghz_analytical
from scipy.optimize import minimize
import copy
import multiprocessing as mp
import sage.all as sg
import numpy as np
import pickle 
############################################### Classes ##########################################################


class Simulation():

    def __init__(self,load):
        self.setup_char = "O-O-x-O-"
        if load:
            print('Loading object O-O-x-O- ')    
            with open("saved_objects/O-O-x-O-.pkl", 'rb') as inp:
                self.setup = pickle.load(inp)
        else:
            self.setup = system.system(self.setup_char,MMA=True,ManyVariables=True,TwoPhotonResonance= True)
        self.variables_declaration()
        self.equate_detunings_for_symmetry()
        self.create_parameter_dict()
        print('Preparing Analytical sub-class')
        self.Analytical = ghz_analytical.Analytical(self)
        print('\nDone!')
        
        
    def create_parameter_dict(self):
        '''
        Creates a global parameter setting which shall be used for the simulations. 
        The free parameters shall be afterwards the cooperativities and the detunings.
        '''  
        self.parameters = dict()
        # Variables that will be substituted
        self.variables = ['v','g','g_f','gamma','gamma_g','gamma_f','phi','Omega','kappa_c','kappa_b','g0','gamma0']
        
        gamma_val = 1
        kappa_val = 100*gamma_val
        C = sg.var('C')
        c = sg.var('c')
        g_val = sg.sqrt( C * kappa_val * gamma_val)

        self.parameters =  {sg.var('gamma') : 1,
                            sg.var('gamma0') : gamma_val,
                            sg.var('gamma_g') : 0,
                            sg.var('gamma_f') : 1 * gamma_val,
                            sg.var('kappa_b') : kappa_val,
                            sg.var('kappa_c') : kappa_val,
                            sg.var('g'): g_val,
                            sg.var('g_f') : g_val,
                            sg.var('g0')  : g_val,
                            sg.var('Omega') :  sg.sqrt(C) * gamma_val * 0.25 / 2,
                            sg.var('phi') : 0,
                            sg.var('v')   : sg.sqrt( c * kappa_val * kappa_val) 
                            }



        self.realistic_parameters = copy.deepcopy(self.parameters)
        gamma_g_real = 0.05 * gamma_val
        gamma_f_real = 0.95 * gamma_val
        self.realistic_parameters[sg.var('gamma_g')] = gamma_g_real
        self.realistic_parameters[sg.var('gamma_f')] = gamma_f_real

    def equate_detunings_for_symmetry(self):
        '''
        Substitute detunings so that the setup is symmetric.
        '''
        symmetrical_substitution = {sg.var('De3'): sg.var('De2'),sg.var('De03'): sg.var('De02')}
        
        self.setup.eff_hamiltonian = self.setup.eff_hamiltonian.subs(symmetrical_substitution)
        self.setup.eff_hamiltonian_gs = self.setup.eff_hamiltonian_gs.subs(symmetrical_substitution)

        for lind_op in range(self.setup.number_of_lindblads):
            self.setup.eff_lindblad_list[lind_op] = self.setup.eff_lindblad_list[lind_op].subs(symmetrical_substitution)




    def variables_declaration(self):
        sg.var('DEg',domain='positive',  latex_name =r'\Delta_{E\gamma}')
        sg.var('Deg',domain='positive',  latex_name =r'\Delta_{e\gamma}')
        sg.var('De1','De2',"De3",domain='positive')
        sg.var('De01','De02',"De03",domain='positive')
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
