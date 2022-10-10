from copy import deepcopy
import sage.all as sg
import pandas as pd
import numpy as np
from collections.abc import MutableMapping
from os.path import exists
directory_to_save = 'saved_objects/optimization_results/'



def conditionally_append_result(result_dict):
    '''
    Conditionally appends result, potentially substituting a result of higher cost.
    '''
    flattened_res_dict = flatten_dict(result_dict)
    keys = list(flattened_res_dict.keys())
    ghz_dim = flattened_res_dict["opt_settings.ghz_dim"]

    hardware_keys = []
    for key in keys:
        if key.startswith('hardware'):
            hardware_keys.append(key)
    opt_settings_to_be_the_same = ["opt_settings.fidelity_cap","opt_settings.swap_into_memory","opt_settings.confidence_interval" ]

    keys_to_be_the_same = opt_settings_to_be_the_same  + hardware_keys

    results_df = load_results(ghz_dim,flattened_res_dict)

    bool_df = np.ones(results_df.shape[0], dtype=bool)
    for key  in keys_to_be_the_same:
        bool_df = bool_df * (results_df[key] ==  flattened_res_dict[key])

    

    index_of_prev_entry = results_df[bool_df].index

    if len(index_of_prev_entry) == 0:
        #element does not exist. append it
        results_df = pd.concat([results_df , pd.DataFrame([flattened_res_dict])] ,ignore_index=True)
    elif len(index_of_prev_entry) == 1:
        previous_cost = results_df.loc[index_of_prev_entry[0]]["cost"]
        new_cost = flattened_res_dict["cost"]
        if new_cost < previous_cost:
            results_df = results_df.drop(index_of_prev_entry) 
            results_df = pd.concat([results_df , pd.DataFrame([flattened_res_dict])] ,ignore_index=True)
    
    save_dataframe(results_df,ghz_dim)


def retrieve_result(k,max_split,fidelity_cap,ghz_dim):
    '''
    Retrieves result for a specific simulation.
    '''
    
    results_df = load_results(ghz_dim)

    dict_to_match = {'hardware.k': k ,'hardware.D_max':max_split,'opt_settings.fidelity_cap':fidelity_cap}
    bool_df = np.ones(results_df.shape[0], dtype=bool)
    for key  in dict_to_match:
        bool_df = bool_df * (results_df[key] ==  dict_to_match[key])

    location_in_df = results_df[bool_df].index[0]

    flattened_dict = results_df.loc[location_in_df].to_dict()

    unflattened_dict = unflatten(flattened_dict)

    # change some entries to sg.var for consistency
    sg_dict_unflattened_dict = deepcopy(unflattened_dict)
    
    # make keys of 'hardware' and 'tuning' into sage variables
    entries_to_change_to_sg = ['hardware','tuning']
    sg_dict_unflattened_dict = cast_str_variables_to_sage(sg_dict_unflattened_dict,entries_to_change_to_sg)

    return sg_dict_unflattened_dict

def cast_sage_variables_to_str(dict_to_cast,entries_of_keys):
    '''
    Takes a dictionary and substitutes all elements of the subdictionaries entries_to_cast (which is a list) with a string instead of sage variables.
    '''
    result_dict = deepcopy(dict_to_cast)
    for entry_dict in entries_of_keys:
        del result_dict[entry_dict]
        result_dict[entry_dict] = {}
        for key in dict_to_cast[entry_dict]:
            result_dict[entry_dict][str(key)] = dict_to_cast[entry_dict][key]
    return result_dict

def cast_str_variables_to_sage(dict_to_cast,entries_of_keys):
    '''
    Takes a dictionary and substitutes all elements of the subdictionaries entries_to_cast (which is a list) with a sage variables instead of strings.
    '''
    result_dict = deepcopy(dict_to_cast)
    for entry_dict in entries_of_keys:
        del result_dict[entry_dict]
        result_dict[entry_dict] = {}
        for key in dict_to_cast[entry_dict]:
            result_dict[entry_dict][sg.var(key)] = dict_to_cast[entry_dict][key]
    return result_dict

def flatten_dict(d: MutableMapping, sep: str= '.') -> MutableMapping:
    '''
    Flatten dict
    https://www.freecodecamp.org/news/how-to-flatten-a-dictionary-in-python-in-4-different-ways/
    '''
    [flat_dict] = pd.json_normalize(d, sep=sep).to_dict(orient='records')
    return flat_dict

def save_dataframe(dataframe,ghz_dim):
    '''
    Saves dataframe in the specified folder.
    '''
    dataframe.to_csv(directory_to_save+f'results_ghz_{ghz_dim}.csv', index=False)


def load_results(ghz_dim,flattened_res_dict= None):
    '''
    Loads dataframe from the specified folder.
    Flattened result is used to initialize dataframe in case the file does not exist.
    '''
    directory_of_file = directory_to_save+f'results_ghz_{ghz_dim}.csv'
    if not exists(directory_of_file):
        save_dataframe(pd.DataFrame([flattened_res_dict]),ghz_dim)
    return pd.read_csv(directory_of_file)




def unflatten(dictionary):
    '''
    https://stackoverflow.com/a/6037657
    '''
    resultDict = dict()
    for key, value in dictionary.items():
        parts = key.split(".")
        d = resultDict
        for part in parts[:-1]:
            if part not in d:
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return resultDict