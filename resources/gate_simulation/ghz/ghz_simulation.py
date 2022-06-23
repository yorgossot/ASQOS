#
# File containing all necessary classes to simulate the gate through jupyter notebooks.
#

from ...system import system
from . import ghz_analytical
from . import bell_pair_superoperator
from scipy.optimize import minimize
import copy, pickle
import multiprocessing as mp
import sympy as sp
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
        self.variables = ['v','g','g_f','gamma','gamma_g','gamma_f','phi','Omega','kappa_c','kappa_b','g_0','gamma_0']
        
        
        
        C = sp.Symbol('C')
        c = sp.Symbol('c')
        kappa_c_val = 100 * C
        kappa_b_val = 100*kappa_c_val*c
        g_val = sp.sqrt( C * kappa_c_val )

        self.parameters =  {sp.Symbol('gamma') : 1,
                            sp.Symbol('gamma_0') : 1,
                            sp.Symbol('gamma_g') : 0,
                            sp.Symbol('gamma_f') : 1 ,
                            sp.Symbol('kappa_b') : kappa_b_val,
                            sp.Symbol('kappa_c') : kappa_c_val,
                            sp.Symbol('g'): g_val,
                            sp.Symbol('g_f') : g_val,
                            sp.Symbol('g_0')  : g_val,
                            sp.Symbol('Omega') :  sp.sqrt(C)  * 0.25 / 2,
                            sp.Symbol('phi') : 0,
                            sp.Symbol('nu')   : sp.sqrt( c * kappa_c_val * kappa_b_val) 
                            }



        self.realistic_parameters = copy.deepcopy(self.parameters)
        gamma_g_real = 0.05 
        gamma_f_real = 0.95 
        self.realistic_parameters[sp.Symbol('gamma_g')] = gamma_g_real
        self.realistic_parameters[sp.Symbol('gamma_f')] = gamma_f_real


