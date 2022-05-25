#
# File containing all necessary classes to simulate the gate through jupyter notebooks.
#

from ...system import system
from . import ghz_analytical
from . import bell_pair_superoperator
from scipy.optimize import minimize
import copy, pickle
import multiprocessing as mp
import sage.all as sg
import numpy as np
############################################### Classes ##########################################################


class Simulation():

    def __init__(self,setup_char,load_setup = True, load_analytical=True):
        self.setup_char = setup_char
        self.load_analytical = load_analytical
        if load_setup == True: 
            print(f"Loading {setup_char} setup")
            # Load from saved object to save time and memory.   
            with open(f'saved_objects/setups/{setup_char}.pkl', 'rb') as inp:
                self.setup = pickle.load(inp)           
        else:
            self.setup = system.system(setup_char,MMA=True,ManyVariables=False,TwoPhotonResonance= True)
        self.variables_declaration()
        self.create_parameter_dict()
        print('Preparing Analytical sub-class')
        self.Analytical = ghz_analytical.Analytical(self)
        print('Preparing SuperoperatorBellPair sub-class')
        self.SuperoperatorBellPair = bell_pair_superoperator.Superoperator(self)
        print('\nDone!')
        
        
    def create_parameter_dict(self):
        '''
        Creates a global parameter setting which shall be used for the simulations. 
        The free parameters shall be afterwards the cooperativities and the detunings.
        '''  
        self.parameters = dict()
        # Variables that will be substituted
        self.variables = ['v','g','g_f','gamma','gamma_g','gamma_f','phi','Omega','kappa_c','kappa_b','g0','gamma0']
        
        
        
        C = sg.var('C')
        c = sg.var('c')
        kappa_c_val = 100 * C
        kappa_b_val = 100*kappa_c_val*c
        g_val = sg.sqrt( C * kappa_c_val )

        self.parameters =  {sg.var('gamma') : 1,
                            sg.var('gamma0') : 1,
                            sg.var('gamma_g') : 0,
                            sg.var('gamma_f') : 1 ,
                            sg.var('kappa_b') : kappa_b_val,
                            sg.var('kappa_c') : kappa_c_val,
                            sg.var('g'): g_val,
                            sg.var('g_f') : g_val,
                            sg.var('g0')  : g_val,
                            sg.var('Omega') :  sg.sqrt(C)  * 0.25 / 2,
                            sg.var('phi') : 0,
                            sg.var('v')   : sg.sqrt( c * kappa_c_val * kappa_b_val) 
                            }



        self.realistic_parameters = copy.deepcopy(self.parameters)
        gamma_g_real = 0.01 
        gamma_f_real = 0.99 
        self.realistic_parameters[sg.var('gamma_g')] = gamma_g_real
        self.realistic_parameters[sg.var('gamma_f')] = gamma_f_real


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
