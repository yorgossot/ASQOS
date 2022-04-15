#
## Library Containing functions to simulate GHZ generation via Monte Carlo, sampling or infinite sums.
#
import numpy as np
import itertools, tqdm
import multiprocessing
import time 

maximum_allowed_time = 10**7


def sampling_function_GHZ(mc_trials,probability_of_success):
    frequency_array = np.zeros(maximum_allowed_time,dtype=int)

    # Create random generator of the thread
    rand_int_of_thread = int( multiprocessing.current_process()._identity[0] * time.time()*100000 )
    thread_random_generator = np.random.default_rng(seed=rand_int_of_thread)

    ghz_achieved_in = thread_random_generator.geometric(p=probability_of_success, size=mc_trials)

    links_created = np.sum(ghz_achieved_in) 
    linkA = thread_random_generator.geometric(p=probability_of_success,size=links_created)
    linkB = thread_random_generator.geometric(p=probability_of_success,size=links_created)

    t_link = np.maximum(linkA,linkB)
    t_link_divided_list = np.array(np.split(t_link,np.cumsum(ghz_achieved_in)[0:-1]))

    for trial_ind in range(mc_trials):
        ghz_time = ghz_achieved_in[trial_ind]
        links_time = np.sum(t_link_divided_list[trial_ind])
        frequency_array[ ghz_time+links_time] += 1 
    return frequency_array


def multiprocessed_sampling_ghz(probability_of_success, mc_trials, number_of_threads, repeat=1 ):

    trials_per_thread = int(mc_trials/number_of_threads)

    frequency_array = np.zeros(maximum_allowed_time,dtype=int)

    for i in range(repeat):
        with mp.Pool(number_of_threads) as pool:
            parameters = itertools.repeat((trials_per_thread,probability_of_success),number_of_threads)
            for result_arr in tqdm.tqdm(pool.istarmap(sampling_function_GHZ, parameters),total=number_of_threads):
                frequency_array += result_arr


    frequency_array = np.trim_zeros(frequency_array,trim='b')

    sampled_probability_distribution = frequency_array/np.sum(frequency_array)

    return sampled_probability_distribution





def sampling_function_GHZ_test(mc_trials,probability_of_success):
    frequency_array = np.zeros(500,dtype=int)
    ghz_2d = np.zeros((500,500),dtype=int)

    # Create random generator of the thread
    rand_int_of_thread = int( multiprocessing.current_process()._identity[0] * time.time()*100000 )
    thread_random_generator = np.random.default_rng(seed=rand_int_of_thread)

    ghz_achieved_in = thread_random_generator.geometric(p=probability_of_success, size=mc_trials)

    links_created = np.sum(ghz_achieved_in) 
    linkA = thread_random_generator.geometric(p=probability_of_success,size=links_created)
    linkB = thread_random_generator.geometric(p=probability_of_success,size=links_created)

    t_link = np.maximum(linkA,linkB)
    t_link_divided_list = np.array(np.split(t_link,np.cumsum(ghz_achieved_in)[0:-1]))

    t_diff = np.abs(linkA-linkB)
    t_diff_divided_list = np.array(np.split(t_link,np.cumsum(ghz_achieved_in)[0:-1]))

    for trial_ind in range(mc_trials):
        ghz_time = ghz_achieved_in[trial_ind]
        links_time = np.sum(t_link_divided_list[trial_ind])

        
        frequency_array[ ghz_time+links_time] += 1 

        ghz_2d[ghz_time+links_time][ t_diff_divided_list[trial_ind][-1] ] += 1
    return frequency_array, ghz_2d


def multiprocessed_sampling_ghz_test(probability_of_success, mc_trials, number_of_threads, repeat=1 ):

    trials_per_thread = int(mc_trials/number_of_threads)

    frequency_array = np.zeros(500,dtype=int)
    ghz_2d = np.zeros((500,500),dtype=int)
    for i in range(repeat):
        with mp.Pool(number_of_threads) as pool:
            parameters = itertools.repeat((trials_per_thread,probability_of_success),number_of_threads)
            for result_arr1,result_arr2 in tqdm.tqdm(pool.istarmap(sampling_function_GHZ_test, parameters),total=number_of_threads):
                frequency_array += result_arr1
                ghz_2d += result_arr2


    frequency_array = np.trim_zeros(frequency_array,trim='b')

    sampled_probability_distribution = frequency_array/np.sum(frequency_array)

    for i in range(500):
        if np.sum(ghz_2d[i][:]):
            ghz_2d[i][:] =  ghz_2d[i][:]/np.sum(ghz_2d[i][:])
        
    return sampled_probability_distribution ,ghz_2d





import multiprocessing as mp


def multiprocessed_monte_carlo(probability_of_creating_link, mc_trials, number_of_threads):
    '''
    Function that runs monte carlo experiments and returns the probability distribution.
    '''
    successes = np.zeros(maximum_allowed_time)
    # Multi-threading the operation
    with mp.Pool(number_of_threads) as pool:
        parameters = itertools.repeat((probability_of_creating_link,),mc_trials)
        for time_to_generate_GHZ in tqdm.tqdm(pool.istarmap(run_monte_carlo_trial, parameters),total=mc_trials):
            successes[ time_to_generate_GHZ ] += 1
    
    successes = np.trim_zeros(successes, 'b') # get rid of final zero elements
    success_probability_distribution = successes / mc_trials
    
    return success_probability_distribution


def run_monte_carlo_trial(probability_of_creating_link):
    '''
    Function that does 4-GHZ state generation and returns the time needed for this generation.

    Parameters:
        probability_of_creating_link : float
            Probability of succesfully creating an entangled link or combining the links.
        decoherence_time : float
            Time it takes for entangled link to decohere in units of gate time.
    Returns:
        time : int
            Time needed to generate the GHZ state in units of gate time.
    '''
    time = 0
    number_of_links = 2
    link = [{'id':i , 'is_entangled':False } for i in range(number_of_links)]
    GHZ4_is_generated  = False
    while not GHZ4_is_generated:
        all_links_are_entangled = True
        for i in range(number_of_links):
            all_links_are_entangled = all_links_are_entangled and link[i]['is_entangled'] 

        if not all_links_are_entangled:
            # Attempt to entangle all links
            for i in range(number_of_links):
                # Iterate through the 2 links
                if not link[i]['is_entangled']:
                    # If it is not entangled, attempt to entangle. If successful reset the decoherence time
                    if np.random.random() < probability_of_creating_link:
                        link[i]['is_entangled'] = True
        else:
            # Attempt GHZ generation
            if np.random.random() < probability_of_creating_link:
                GHZ4_is_generated = True
            else:
                # If it fails, reset the links.
                for i in range(number_of_links):
                    link[i]['is_entangled'] = False
        time += 1
    return int(time)


def run_monte_carlo_trial_with_decoherence_cutoff(probability_of_creating_link,decoherence_time):
    '''
    Function that does 4-GHZ state generation and returns the time needed for this generation.

    Parameters:
        probability_of_creating_link : float
            Probability of succesfully creating an entangled link or combining the links.
        decoherence_time : float
            Time it takes for entangled link to decohere in units of gate time.
    Returns:
        time : int
            Time needed to generate the GHZ state in units of gate time.
    '''
    time = 0
    number_of_links = 2
    link = [{'id':i , 'is_entangled':False , 'time_left_until_decoherence':decoherence_time} for i in range(number_of_links)]
    GHZ4_is_generated  = False
    while not GHZ4_is_generated:
        all_links_are_entangled = True
        for i in range(number_of_links):
            all_links_are_entangled = all_links_are_entangled and link[i]['is_entangled'] 

        if not all_links_are_entangled:
            # Attempt to entangle all links
            for i in range(number_of_links):
                # Iterate through the 2 links
                if not link[i]['is_entangled']:
                    # If it is not entangled, attempt to entangle. If successful reset the decoherence time
                    if np.random.random() < probability_of_creating_link:
                        link[i]['is_entangled'] = True
                        link[i]['time_left_until_decoherence'] = decoherence_time
                else:
                    # If it is entangled, check for decoherence.
                    link[i]['time_left_until_decoherence'] -= 1
                    if link[i]['time_left_until_decoherence'] <= 0:
                        link[i]['is_entangled'] = False
                # If the link is not entangled, the flag will be False afterwards
        else:
            # Attempt GHZ generation
            if np.random.random() < probability_of_creating_link:
                GHZ4_is_generated = True
            else:
                # If it fails, reset the links.
                for i in range(number_of_links):
                    link[i]['is_entangled'] = False
        time += 1
    return int(time)

