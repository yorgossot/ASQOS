import numpy as np
import itertools , math , time



def ps(n,probability_of_success):
    '''
    Success after n failed trials.
    '''
    return probability_of_success*(1-probability_of_success)**n

#
# Obtain  
#

max_nn_array = 100
nn_array = np.zeros((max_nn_array,max_nn_array))
for (ind0,ind1), _ in np.ndenumerate(nn_array):
    nn_array[ind0,ind1] = ps(ind0)*ps(ind1)


nn_array_ind = np.dstack(np.unravel_index(np.argsort(nn_array.ravel()), np.shape(nn_array)))[0][::-1]

nn_array = np.sort(nn_array.flatten())[::-1]
'''
elements_to_delete = []
for (index,element) in enumerate(nn_array_ind):
    n_i , n_j = tuple(element)
    if abs(n_i - n_j) >= decoherence_time:
        elements_to_delete.append(index)

nn_array = np.delete(nn_array,elements_to_delete)
nn_array_ind = np.delete(nn_array_ind,elements_to_delete,axis=0)
'''
nn_array_time = np.zeros_like(nn_array)
for (i,succ_index) in enumerate(nn_array_ind):
    nn_array_time[i] = max(succ_index) +1

max_time_link = int( np.max(nn_array_time) )

p_link_array = np.zeros(max_time_link+1)
mult_link_array = np.zeros(max_time_link+1)
for (ind,tim) in enumerate(nn_array_time):
    p_link_array[int(tim)] += nn_array[ind]
    mult_link_array[int(tim)] +=1 

decimal_tolerance = 7
p_link_array_cum_sum = np.cumsum(p_link_array)

p_link_array_cutoff = np.where(np.around(p_link_array_cum_sum,decimals=decimal_tolerance)==1)[0][0]

p_link_array = p_link_array[0:p_link_array_cutoff]
mult_link_array = p_link_array[0:p_link_array_cutoff]



def C_temp_function(which_nn_to_take,number_of_fails=0):
    number_of_successes = len(which_nn_to_take)
    which_nn_to_take.sort()
    divided_into_sets = [list(grp) for _, grp in itertools.groupby(which_nn_to_take)]
    
    number_of_types_of_success =len(divided_into_sets) 
    success_multiplicities = np.zeros(number_of_types_of_success)
    for i in range(number_of_types_of_success):
        success_multiplicities[i] = len(divided_into_sets[i])
        
    fact = math.factorial(number_of_successes + number_of_fails -1)

    ans = 0
    for i in range(number_of_types_of_success):
        contribution = fact
        for (j,mult) in enumerate(success_multiplicities):
            if j == i:
                contribution /= math.factorial(mult-1)
            else:
                contribution /= math.factorial(mult)
        ans += contribution

    return int(ans)

def partition_min_max(n,k,l,m):
    '''
    n is the integer to partition, k is the length of partitions, 
    l is the min partition element size, m is the max partition element size
    https://stackoverflow.com/a/66389282
    '''
    if k < 1:
        return
    if k == 1:
        if n <= m and n>=l :
            yield (n,)
        return
    if (k*128) < n: #If the current sum is too small to reach n
        return
    if k*1 > n:#If current sum is too big to reach n
        return
    for i in range(l,m+1):
        for result in partition_min_max(n-i,k-1,i,m):                
            yield result+(i,)


def partition(N, size):
    n = N + size - 1
    for splits in itertools.combinations(range(n), size - 1):
        yield [s1 - s0 - 1 for s0, s1 in zip((-1,) + splits, splits + (n,))]


def integer_partition(n,size):
    '''
    Partition integer n in specific size partitions
    '''
    return partition_min_max(n,size,1,n) #partition(n,size)#


def increment_list_t(which_nn_to_take, previous_integer_partition , previous_index):
    incremented_flag = False

    current_total_time = sum(which_nn_to_take)  
    
    size_of_partitions = len(which_nn_to_take)

    partitions_of_time = previous_integer_partition

    if previous_index + 1 == len(partitions_of_time):
        new_index = 0
        current_total_time +=1
        new_integer_partition = [ i for i in integer_partition(current_total_time,size_of_partitions) ]
        new_which_nn_to_take = list( new_integer_partition[0] )    
    else:
        new_index = previous_index + 1
        new_which_nn_to_take = list( partitions_of_time[new_index] )
        new_integer_partition = previous_integer_partition
    
    return new_which_nn_to_take , new_integer_partition , new_index


# Analytical according to thesis notes
number_of_threads = 1

probability_t = np.zeros(10**5)

sumtest = 0

tolerance = 10**(-5)
m = 0 #GHZ failed attempts before success
current_tolerance = 1
tic = time.time()
continue_bool = True
while m<20 and np.sum(probability_t)<0.99:
    print(f'm = {m}')
    p_ghz = ps(m)

    ghz_comb_time = m + 0

    # Successful creations before  are m
    which_nn_to_take = [1 for i in range(m+1)]

    current_link_time = m+1
    size_of_partitions = m+1
    new_integer_partition = [ i for i in integer_partition(current_link_time,size_of_partitions) ]
    new_index = 0

    continue_bool_m = True
    tic = time.time()
    while continue_bool_m:
        
        which_nn_to_take_list_for_mp = [tuple(which_nn_to_take)]

        
        #for i in range(number_of_threads):
        #    which_nn_to_take, new_integer_partition, new_index = increment_list_t(which_nn_to_take,new_integer_partition, new_index)
        #    which_nn_to_take_list_for_mp.append(tuple(which_nn_to_take))

        def function_to_multiprocess(which_nn_to_take,**kwargs):
            
            which_nn_to_take = list(which_nn_to_take)

            success_product = 1
            success_time = 0  
            min_which_nn = max_nn_array
            for which_nn in which_nn_to_take:
                success_product = success_product * p_link_array[which_nn] #* mult_link_array[which_nn]
                time_it_took = which_nn
                success_time += time_it_took

            '''
            fail_tolerance_flag = True
            fail_index = 0
            while fail_tolerance_flag:
                C_temp =  C_temp_function(which_nn_to_take , 0)
                total_probability =  fail_array[fail_index] * success_product * p_ghz * C_temp
                fail_time = fail_index * decoherence_time
                total_time = success_time + fail_time + ghz_comb_time
                if total_probability > tolerance:
                    probability_t[total_time] += total_probability
                else:
                    fail_tolerance_flag = False
                fail_index += 1
            '''
            C_temp =  C_temp_function(which_nn_to_take , 0)

            total_probability =   success_product * p_ghz * C_temp
            fail_time = 0
            total_time = success_time + fail_time + ghz_comb_time

            return total_probability , total_time
        
        '''
        with mp.Pool(number_of_threads) as pool:
            parameters = zip(which_nn_to_take_list_for_mp)
            
            for total_probability , total_time in pool.istarmap(function_to_multiprocess, parameters):
                probability_t[total_time] += total_probability
        '''
        for (which_nn) in which_nn_to_take_list_for_mp:
            # No multiprocessing
            total_probability , total_time = function_to_multiprocess(which_nn)
            probability_t[total_time] += total_probability

        which_nn_to_take, new_integer_partition, new_index  = increment_list_t(which_nn_to_take, new_integer_partition, new_index  ) #, continue_bool_m = increment_list(which_nn_to_take,decoherence_time )
        #print(max(which_nn_to_take))
        if max(which_nn_to_take)>len(p_link_array)-2 or time.time()-tic>2:
    
            tic = time.time()
            break
    
    m+=1