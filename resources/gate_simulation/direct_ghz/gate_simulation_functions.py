
import math, json
import copy
import numpy as np
import sage.all as sg
import qutip as qt
from functools import partial

###############################################  Parameters  ##########################################################


###############################################  Import data  ##########################################################


with  open('resources/experimental_values.json') as json_file: 
    experimental_values_dict = json.load(json_file) 

###############################################  Performance assessment  ##########################################################


def gate_performance_cost_function(performance_dict, opt_settings_dict):
    '''
    Cost function to be minimized in order to achieve maximum performance.
    '''
    maximum_cost = 100*opt_settings_dict['fidelity_cap']
    #when the fidelity cap is achieved, the cost will be 0

    if performance_makes_sense(performance_dict):
        
        if performance_dict['fidelity'] >= opt_settings_dict['fidelity_cap']:
            t_conf, _ = time_interval_of_confidence(opt_settings_dict,performance_dict)
            cost = maximum_cost - 100*opt_settings_dict['fidelity_cap'] - 1/ t_conf
        else:
            cost = maximum_cost - 100*performance_dict['fidelity']
    else:
        # if the performance parameters dont make sense, give the maximum cost 
        cost = maximum_cost
    return cost
    
def performance_makes_sense(performance_dict):
    '''
    Checks if performance parameters make sense.
    '''

    performance_array = np.array([performance_dict['fidelity'],performance_dict['p_success'],performance_dict['gate_time']])
    if np.any(np.zeros(3) >= performance_array):
        return False
    if np.any(np.full_like(performance_array,np.inf) == performance_array):
        return False
    if performance_dict['fidelity'] > 1.001 or performance_dict['p_success'] > 1.001:
        return False
    return True


############################################### t_conf _functions  #############################

def time_interval_of_confidence(opt_settings_dict, performance_dict ):
    '''
    Returns the number of attempts to achieve success for a geometric distribution given a confidence interval.
    '''
     
    if opt_settings_dict['swap_into_memory'] == False:
        memory_bool = opt_settings_dict['swap_into_memory']
        t_conf = time_interval_of_confidence_without_memory(opt_settings_dict,performance_dict)
    else:
        raise Exception("Sorry, set swap into memory to False")
    
    return t_conf , memory_bool

def effective_gate_time(gate_time):
    '''
    Calculates and returns the effective gate time according to hyperpaprameters.
    '''
     # t_measure to reset the qubits by measuring and t_Hel to intialize them
    time_to_reset = experimental_values_dict['t_measure'] + experimental_values_dict["t_H_el"]
    # Gate time is gate_time + t_measure + t_reset 
    gate_time_in_s =  gate_time/experimental_values_dict['gamma'] +experimental_values_dict['t_measure'] + time_to_reset
    return gate_time_in_s


def time_interval_of_confidence_without_memory(opt_settings_dict,performance_dict):
    
    gate_time_in_s =  effective_gate_time(performance_dict['gate_time'])

    bell_pairs_to_create = 1
    #Geometric distribution with probability: ghz dimension-1: number of pairs to create
    p_succ =  performance_dict['p_success']**(bell_pairs_to_create)  
    if 1-p_succ < 1:
        t_conf =  gate_time_in_s*math.ceil( math.log(1-opt_settings_dict['confidence_interval']) / math.log(1-p_succ) )
        t_conf += experimental_values_dict["t_swap_el_nuc"]
    else:
        t_conf = 10**24
    return t_conf







### Utilities for qubit operations ########################3#


def ten_r(rot_matr,qubit,n_qubits):
    '''
    Works out the tensor product of U in a n_qubit level system if U is diagonal.
    '''
    ten_matr = sg.Matrix(sg.SR,np.zeros((1,1)))
    ten_matr[0,0] = 1
    
    I = sg.identity_matrix(2)
    for i in range(n_qubits):
        if i ==qubit:
            ten_matr = ten_matr.tensor_product(rot_matr)
        else:
            ten_matr = ten_matr.tensor_product(I)
    return ten_matr

def R_z(theta):
    '''
    Rz parametric qubit gate as sagemath matrix
    '''
    r_matr = sg.Matrix(sg.SR,np.zeros((2,2)))
    r_matr[0,0] = 1
    r_matr[1,1] = sg.exp(sg.I*theta)
    return r_matr


def R_y(theta):
    '''
    Ry parametric qubit gate as sagemath matrix
    '''
    r_matr = sg.Matrix(sg.SR,np.zeros((2,2)))
    r_matr[0,0] = sg.cos(theta/2)
    r_matr[1,1] = sg.cos(theta/2)
    r_matr[0,1] = -sg.sin(theta/2)
    r_matr[1,0] = sg.sin(theta/2)
    return r_matr




def GHZ_symbolic_fidelity_from_evolution(evolution,rotation):
    '''
    Returns the fidelity of a 4-GHZ state given a unitary when that is in symbolic form
    '''
    n_qubits = 3
    GHZ_state = np.zeros(2**n_qubits)
    GHZ_state[[0,-1]] = 1/np.sqrt(2)
    GHZ_state_row_vec = sg.vector(sg.SR,GHZ_state).row()


    rots = []
    for i in range(2): rots.append(sg.var(f'r{i}_r')*rotation[i])  #direct GHZ tweaked  
    
    R = []
    for rot_val in rots:
        r_matr = sg.Matrix(sg.SR,np.zeros((2,2)))
        r_matr[0,0] = 1
        r_matr[1,1] = sg.exp(sg.I*rot_val)
        R.append(r_matr)

    H = sg.Matrix( qt.qip.operations.hadamard_transform(1).data.toarray() )
    plus_state = qt.Qobj(np.array([1,1])/np.sqrt(2) )
    init_state = qt.tensor(*(plus_state for i in range(n_qubits)))



    current_state = sg.vector(init_state.data.toarray().reshape(2**n_qubits)).column()

    current_state = evolution*current_state

    current_state = ten_r(R[1],0,n_qubits)*ten_r(R[0],2,n_qubits)*ten_r(R[0],1,n_qubits)*current_state

    for i in range(3): current_state =   ten_r(H,i,n_qubits) *current_state
        
    fidelity_sg =  sg.abs_symbolic( (GHZ_state_row_vec * current_state)[0][0]  ) **2 

    return fidelity_sg

