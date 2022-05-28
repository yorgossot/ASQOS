
from cgitb import lookup
import math, json
import copy
import numpy as np
import sage.all as sg
import qutip as qt
from functools import partial

###############################################  Parameters  ##########################################################


###############################################  Import data  ##########################################################
p_lookup_table_ghz = {}
with open(f'resources/gate_simulation/ghz/lookup_tables_negative_binomial/ghz{3}_table_conf0.99.npy', 'rb') as f:
    p_lookup_table_ghz[3] = np.load(f)

with open(f'resources/gate_simulation/ghz/lookup_tables_negative_binomial/ghz{4}_table_conf0.99.npy', 'rb') as f:
    p_lookup_table_ghz[4] = np.load(f)

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
     
    if opt_settings_dict['swap_into_memory'] == True:
        memory_bool = opt_settings_dict['swap_into_memory']
        t_conf = time_interval_of_confidence_with_memory(opt_settings_dict,performance_dict)
    elif opt_settings_dict['swap_into_memory'] == False:
        memory_bool = opt_settings_dict['swap_into_memory']
        t_conf = time_interval_of_confidence_without_memory(opt_settings_dict,performance_dict)
    elif opt_settings_dict['swap_into_memory'] == None:
        t_confs = [time_interval_of_confidence_without_memory(opt_settings_dict,performance_dict),
                    time_interval_of_confidence_with_memory(opt_settings_dict,performance_dict)]
        argmin = np.argmin(t_confs)
        t_conf = t_confs[argmin]
        if argmin == 0:
            memory_bool = False
        else:
            memory_bool = True
    
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

    bell_pairs_to_create = opt_settings_dict["ghz_dim"] - 1
    #Geometric distribution with probability: ghz dimension-1: number of pairs to create
    p_succ =  performance_dict['p_success']**(bell_pairs_to_create)  
    if 1-p_succ < 1:
        t_conf =  gate_time_in_s*math.ceil( math.log(1-opt_settings_dict['confidence_interval']) / math.log(1-p_succ) )
        t_conf += experimental_values_dict["t_swap_el_nuc"]
    else:
        t_conf = 10**24
    return t_conf


def time_interval_of_confidence_with_memory(opt_settings_dict,performance_dict):
    '''
    Finds time interval of confidence when quantum memory is assumed.
    '''
    gate_time_in_s =  effective_gate_time(performance_dict['gate_time'])

    bell_pairs_to_create = opt_settings_dict["ghz_dim"] - 1

    p_lookup_table = p_lookup_table_ghz[opt_settings_dict["ghz_dim"]]

    attempts = np.searchsorted(-p_lookup_table,-performance_dict['p_success']) #find the first value that is lower than the probability of success
    gates_time = attempts * gate_time_in_s
    swaps_time = (bell_pairs_to_create)*experimental_values_dict["t_swap_el_nuc"]
     
    t_conf = gates_time + swaps_time

    return t_conf

################# concurrence functions ############################

def concurrence_from_ket(ket):
    '''
    Calculates concurrence for 4-ket of numpy.
    '''
    ket = np.array(ket,dtype=complex)
    qobj = qt.Qobj(ket).unit() #normalize
    qobj.dims =  [[2, 2], [1, 1]]
    return qt.concurrence(qobj)


def concurrence_from_evolution(evolution,tuning_dict):
    '''
    Calculates concurrence for a unitary.
    '''
    n_qubits = 2
    evolution = evolution.subs(tuning_dict)
    
    plus_state = qt.Qobj(np.array([1,1])/np.sqrt(2) )
    init_state = qt.tensor(*(plus_state for i in range(n_qubits)))
    init_state = sg.vector(init_state.data.toarray().reshape(2**2)).column()

    # Add initial rotations
    for q in range(n_qubits):
        R = R_y(tuning_dict[sg.var(f'r0_i')])
        init_state = ten_r(R,q,n_qubits=2)*init_state

    final_state = ten_u((0,1),evolution,2)*init_state
       
    final_state = np.array(final_state,dtype=complex)

    return concurrence_from_ket(final_state)



### Utilities for qubit operations ########################3#

def ten_u(pair,evolution,n_qubits):
    '''
    Works out the tensor product of U in a n_qubit level system if U is diagonal.
    '''
    ten_matr = sg.Matrix(sg.SR,np.zeros((2**n_qubits,2**n_qubits)))

    for i in range(2**n_qubits):
        i_bin = "{0:b}".format(i)
        # Add zeros
        i_bin = '0'*(n_qubits-len(i_bin)) + i_bin
        u_bin = i_bin[pair[0]] + i_bin[pair[1]]
        u_el = int(u_bin,2)
        ten_matr[i,i] = evolution[u_el,u_el]
    return ten_matr

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
