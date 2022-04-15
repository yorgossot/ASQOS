# Generates lookup table that corresponds probabilities to some number of attempts.

import numpy as np
from scipy.special import betaincinv

def generate_lookup_table(ghz_dim,conf_interval):
    '''
    Function to generate lookup_table according to negative binomial distribution.
    '''
    bell_pairs_to_create = ghz_dim - 1
    attempts = np.arange(0,10000)
    p = betaincinv(bell_pairs_to_create,attempts+1-bell_pairs_to_create, conf_interval)
    
    with open(f'ghz{ghz_dim}_table_conf{conf_interval}.npy', 'wb') as f: np.save(f,p)