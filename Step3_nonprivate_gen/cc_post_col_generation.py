import pandas as pd 
import os 
import numpy as np 

def sample_from_truth(x, prob_dict, triplet_prob_dict, overall_prob_dict):
    if x in prob_dict.keys():
        ret = np.random.choice(prob_dict[x].index, p = prob_dict[x])
    elif x in triplet_prob_dict.keys():
        ret = np.random.choice(triplet_prob_dict[x].index, p = triplet_prob_dict[x])
    else:
        ret = np.random.choice(list(overall_prob_dict.keys()), p = list(overall_prob_dict.values()) )
    return(ret)

def post_col_generation(public_data_file, triplets_dir, sampling_dir, epsilon):
    # load already created submission files 
    final = pd.read_csv(os.path.join(triplets_dir, 'sampled_triplet_{0}.csv'.format(epsilon)))
    marginal_cols = ['shift', 'pickup_community_area', 'dropoff_community_area']

    final['sp'] = list(zip(final['shift'], final['pickup_community_area'], final['dropoff_community_area']))


    # get distribution of payment type for each location set 
    truth = pd.read_csv(public_data_file)

    sim_cols = [
                'fare',
                'trip_miles',
                'trip_seconds',
                'tips',
                'trip_total',
                'payment_type'
                ]
    
    # get count of each key in final 
    final_count_dict = dict()
    grouped_final = final.groupby(marginal_cols)
    final_count_dict = dict()
    keys = []
    for sp, vals in grouped_final:
        keys.append(sp)
        final_count_dict[sp] = vals.shape[0]

    overall_prob_dicts = dict()
    triplet_prob_dicts = dict()
    prob_dicts = dict()
    for col in sim_cols:
        print('generating dictionaries for {0}'.format(col))
        if col in ['fare']:
            dict_cols = marginal_cols 
        elif col in ['trip_miles', 'trip_seconds']:
            dict_cols = marginal_cols + ['fare']
        elif col in ['tips']:
            dict_cols = marginal_cols + ['fare']
        elif col == 'trip_total':
            dict_cols = ['fare', 'tips']
        elif col == 'payment_type':
            dict_cols = ['fare', 'tips', 'trip_total']
        else:
            dict_cols = marginal_cols
        grouped = truth.groupby(dict_cols)

        # generate overall probability dictionary for col
        overall_counts = truth[col].value_counts(sort = False)
        overall_prob_dict = dict(overall_counts / np.sum(overall_counts))
        overall_prob_dicts[col] = overall_prob_dict

        prob_dict = dict()
        for sp, vals in grouped[col]:
            # find marginal dist
            counts = vals.value_counts(sort = False) 
            probs = counts / np.sum(counts)
            prob_dict[sp] = probs
        prob_dicts[col] = prob_dict

        triplet_prob_dict = dict()
        triplet_grouped = truth.groupby(marginal_cols)
        for sp, vals in triplet_grouped[col]:
            # find marginal dist
            counts = vals.value_counts(sort = False) 
            probs = counts / np.sum(counts)
            triplet_prob_dict[sp] = probs
        triplet_prob_dicts[col] = triplet_prob_dict


    '''
    generate synthetic data
    '''
    keys = final['sp']
    n = final.shape[0]
    for col in sim_cols:
        print('generating data for {0}'.format(col))
        preds = map(lambda key: sample_from_truth(key, prob_dicts[col], triplet_prob_dicts[col], overall_prob_dicts[col]), keys)
        final[col] = list(preds)

    final = final.drop('sp', axis = 1)

    # save updated data
    final.to_csv(os.path.join(sampling_dir, 'final_dataset_{0}.csv'.format(epsilon)), index = False)