import numpy as np 
import pandas as pd 
import os 
import glob
import re
import json
import pathlib
import pickle 
# import analytic_gaussian as ag
import Step1_Archetype_Counting.analytic_gaussian as ag


def get_private_counts(priv_data_path, parameters_path, archetype_path, private_counts_dir, epsilon, delta, archetype_budget_prop):
    # privacy parameters 
    sens = 1
    company_budget_prop = 1 - archetype_budget_prop

    # NOTE: assuming basic composition
    archetype_epsilon = epsilon*archetype_budget_prop
    archetype_delta = delta*archetype_budget_prop
    company_epsilon = epsilon*company_budget_prop
    company_delta = delta*company_budget_prop

    archetype_sigma = ag.calibrateAnalyticGaussianMechanism(archetype_epsilon, archetype_delta, sens, tol = 1.e-12)
    company_sigma = ag.calibrateAnalyticGaussianMechanism(company_epsilon, company_delta, sens, tol = 1.e-12)


    '''load archetype histograms'''
    archetype_dict = dict()
    archetype_files = glob.glob(os.path.join(archetype_path, 'archetype_*.txt'))
    for filename in archetype_files:
        pattern = 'archetype_(.*?).txt'
        archetype_number = int(re.search(pattern, filename).group(1))
        print('creating histogram for archetype {0}'.format(archetype_number))

        # for each archetype...
        with open(filename, 'r') as f:
            # read all lines into vector
            text_vec = f.readlines()

            # create histogram from first 3 lines of file
            hist = []
            for i in range(3):
                elems = [float(elem) for elem in text_vec[i].split(',') if elem != '\n']
                hist.extend(elems)
        
        # store histogram in dictionary
        archetype_dict[archetype_number] = hist

    '''load private data and predict archetypes'''
    print('getting private archetype counts')
    priv_data = pd.read_csv(priv_data_path)
    features = priv_data[ ['shift', 'pickup_community_area', 'dropoff_community_area'] ].to_numpy()
    
    # load gmm
    with open(os.path.join(archetype_path, 'archetypes.pkl'), 'rb') as f:
        gmm = pickle.load(f)

    # predict archetypes for private data 
    priv_data_archetypes = pd.Series(gmm.predict(features))

    # get most likely archetype for each taxi_id
    priv_data['archetype'] = priv_data_archetypes 
    taxi_id_archetypes = priv_data.groupby('taxi_id')['archetype'].agg(lambda x: x.value_counts().index[0])

    '''get archetype counts'''
    priv_data_archetype_counts = taxi_id_archetypes.value_counts(sort = False)
    for arch in archetype_dict.keys():
        if arch not in priv_data_archetype_counts.index:
           priv_data_archetype_counts = priv_data_archetype_counts.append( pd.Series([0], index = [arch]) ) 
    priv_data_archetype_counts = priv_data_archetype_counts.sort_index()
    priv_data_archetype_counts.to_csv(os.path.join(private_counts_dir, f'nonprivate_data_archetype_counts_{epsilon}.csv'), index = False)

    '''get DP archetype counts'''
    final_priv_archetype_counts = priv_data_archetype_counts + np.random.normal(0, archetype_sigma, len(archetype_dict.keys()))
    final_priv_archetype_counts = np.maximum(0, final_priv_archetype_counts)
    priv_count_df = pd.DataFrame({'archetype': list(range(len(priv_data_archetype_counts))),
                                  'count': final_priv_archetype_counts})
    priv_count_df.sort_values('archetype', axis = 0)
    priv_count_df.reset_index(drop = True, inplace = True)
    priv_count_df.to_csv(os.path.join(private_counts_dir, f'private_data_archetype_counts_{epsilon}.csv'), index = False)

    '''get DP company ID counts'''
    print('getting private company ID counts')
    # load company IDs
    with open(parameters_path, 'r') as f:
        parameters_dict = json.loads(pathlib.Path(parameters_path).read_text())
    company_id_vals = list(parameters_dict['schema']['company_id'].values())[2]

    # get most common company ID for each taxi ID, then merge onto original private data 
    top_company_id_modes = priv_data.groupby(['taxi_id'])['company_id'].agg(lambda x: np.random.choice(pd.Series.mode(x))) # could be multimodal
    company_id_counts = top_company_id_modes.value_counts(sort = False)
    for id_val in company_id_vals:
        if id_val not in company_id_counts.index:
            id_val_series = pd.Series([0], index = [id_val])
            company_id_counts = company_id_counts.append( id_val_series )

    # privatize counts 
    priv_company_id_counts = company_id_counts + np.random.normal(0, company_sigma, len(company_id_counts))
    priv_company_id_counts = np.maximum(0, priv_company_id_counts)
    priv_company_count_df = pd.DataFrame({'company_id': priv_company_id_counts.index,
                                        'count': priv_company_id_counts})
    priv_company_count_df.sort_values('company_id', axis = 0, inplace = True)
    priv_company_count_df.reset_index(drop = True, inplace = True)
    priv_company_count_df.to_csv(os.path.join(private_counts_dir, f'private_data_company_id_counts_{epsilon}.csv'), index = False)