from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd 
from collections import defaultdict
import pickle
import os 

def k_archetypes(num_clusters = 10, public_data_file ="public_data.csv"): 

	res_dir = os.path.join('CC_Results_GMM', 'Results_{0}'.format(num_clusters))
	if not os.path.exists(res_dir):
	    os.makedirs(res_dir)

	# Read in datafile
	data = pd.read_csv(public_data_file)

	#seperate out the triplets (pickup_area,dropoff_area,shift) and scale them

	by_shift = pd.pivot_table(
	    data.assign(n=1),
	    values='n',
	    index='taxi_id',
	    columns='shift',
	    aggfunc='count',
	    fill_value=0,
	)
	by_pickup = pd.pivot_table(
	    data.assign(n=1),
	    values='n',
	    index='taxi_id',
	    columns='pickup_community_area',
	    aggfunc='count',
	    fill_value=0,
	)

	by_dropoff = pd.pivot_table(
	    data.assign(n=1),
	    values='n',
	    index='taxi_id',
	    columns='dropoff_community_area',
	    aggfunc='count',
	    fill_value=0,
	)

	counts = by_pickup.join(by_dropoff, rsuffix = 'p').join(by_shift, rsuffix = 'p')

	#Determine clusters of the triplets using the GMM algorithm
	gmm = GaussianMixture(n_components=num_clusters)
	gmm.fit(counts)
	labels = gmm.predict(counts)

	pkl_filename = os.path.join(res_dir, "archetypes.pkl")
	with open(pkl_filename, 'wb') as file:
	    pickle.dump(gmm, file)
	data = data.to_numpy()


	#Get the taxi distributions for each cluster ("Archtype")
	taxi_id = data[0][0]
	count = 0
	overall_count = 0 
	taxi_count = 0
	cluster_counts = np.zeros(num_clusters)
	cluster_lists = [ [] for _ in range(num_clusters) ] 

	for row in data:
		if row[0] != taxi_id and count > 0:
			taxi_count = taxi_count + 1
			# find the cluster this taxi belongs to
			taxi_cluster = np.argmax(cluster_counts)
			cluster_lists[taxi_cluster].append(taxi_id)
			# clear counters for next taxi
			count = 0
			cluster_counts = np.zeros(num_clusters)
			taxi_id = row[0]
		#Determine the cluster belonging to the next trip
		cluster_counts[labels[overall_count]] += 1 
		count += 1
		overall_count = overall_count + 1

	# Print to console information about the clusters
	#for i in range(0,num_clusters):
	#	print("Archetype {}:".format(i))
	#	print("Size = {}".format(len(cluster_lists[i])))
	#	print("Proportion = {}".format(len(cluster_lists[i]) / taxi_count))

	pickup_dists = [ defaultdict(int) for _ in range(num_clusters)]
	dropoff_dists = [ defaultdict(int) for _ in range(num_clusters)]
	shift_dists = [ defaultdict(int) for _ in range(num_clusters)]
	payment_dists =  [ defaultdict(int) for _ in range(num_clusters)]

	taxi_id = data[0][0]
	current_cluster = 0
	# Determine cluster of initial row
	for i in range(num_clusters):
		if data[0][0] in cluster_lists[i]:
			current_cluster = i
			break

	cluster_record_count = np.zeros(num_clusters)
	# Determine the Distributions of the Triplets under each archetype
	for row in data:
		if row[0] != taxi_id:
			for i in range(num_clusters):
				if row[0] in cluster_lists[i]:
					taxi_id = row[0]
					current_cluster = i
					break
		
		pickup_dists[current_cluster][row[3]] += 1
		dropoff_dists[current_cluster][row[4]] += 1
		shift_dists[current_cluster][row[1]] += 1
		payment_dists[current_cluster][row[5]] += 1
		cluster_record_count[current_cluster] += 1

	# Print distributions of Triplets under Archetypes
	#for i in range(0,num_clusters):
	#	print("Archetype {}:".format(i))
	#	print("Shift Dist: {}".format(shift_dists[i]))
	#	print("Pickup Dist: {}".format(pickup_dists[i]))
	#	print("Dropoff Dist: {}".format(dropoff_dists[i]))
	#	print("Payment Dist: {}".format(payment_dists[i]))
				

	# Get normalized triplet distributions per archetype

	pickup_normalized = [ defaultdict(int) for _ in range(num_clusters)]
	dropoff_normalized = [ defaultdict(int) for _ in range(num_clusters)]
	shift_normalized = [ defaultdict(int) for _ in range(num_clusters)]
	payment_normalized = [ defaultdict(int) for _ in range(num_clusters)]

	for i in range(num_clusters):
		pickup_normalized[i] = {k:v/len(cluster_lists[i]) for k,v in pickup_dists[i].items()}
		pickup_normalized[i] = defaultdict(float,pickup_normalized[i])
		dropoff_normalized[i] = {k:v/len(cluster_lists[i]) for k,v in dropoff_dists[i].items()}
		dropoff_normalized[i] = defaultdict(float,dropoff_normalized[i])
		shift_normalized[i] = {k:v/len(cluster_lists[i]) for k,v in shift_dists[i].items()}
		shift_normalized[i] = defaultdict(float,shift_normalized[i])
		payment_normalized[i] = {k:v/len(cluster_lists[i]) for k,v in payment_dists[i].items()}
		payment_normalized[i] = defaultdict(float,payment_normalized[i])


	# Convert distributions to ordered lists, and print them to a file
	# Handle key = -1 seperately
	shift_norm_list = np.zeros((num_clusters,21))
	pickup_norm_list = np.zeros((num_clusters,78))
	dropoff_norm_list = np.zeros((num_clusters,78))
	payment_norm_list = np.zeros((num_clusters,10))

	for i in range(num_clusters):	
		pickup_norm_list[i][0] = pickup_normalized[i][-1.0]
		dropoff_norm_list[i][0] = dropoff_normalized[i][-1.0]
		shift_norm_list[i][0] = shift_normalized[i][0.0]	

	for j in range(1,78):
		for i in range(num_clusters):
			pickup_norm_list[i][j] = pickup_normalized[i][float(j)]
			dropoff_norm_list[i][j] = dropoff_normalized[i][float(j)]
	for j in range(1,21):
		for i in range(num_clusters):
			shift_norm_list[i][j] = shift_normalized[i][float(j)]
	for j in range(-1,9):
		for i in range(num_clusters):
			payment_norm_list[i][j+1] = payment_normalized[i][int(j)]

	# Add labels in dataframe
	df['archetype'] = taxi_column

	# Detemine 3-way marginal distributions per archetype
	archetype_probs = df.groupby('archetype').size().div(len(df))
	within_archetype_3way_marginals = df.groupby(['archetype', 'shift', 'pickup_community_area', 'dropoff_community_area']).size().div(len(df)).div(archetype_probs, axis=0, level='archetype').reset_index()

	# Print marginal distributions to a new file
	within_archetype_3way_marginals.to_csv(os.path.join(res_dir, 'archetype_marginals.csv'))

	# Print archetype information to individual data files
	for i in range(num_clusters):
		f = open(os.path.join(res_dir, "archetype_{}.txt".format(i)),"w")
		for j in range(0,21):
			f.write("{},".format(shift_norm_list[i][j]))
		f.write("\n")
		for j in range(0,78):
			f.write("{},".format(pickup_norm_list[i][j]))
		f.write("\n")
		for j in range(0,78):
			f.write("{},".format(dropoff_norm_list[i][j]))
		f.write("\n")
		for taxi in cluster_lists[i]:
			f.write("{},".format(int(taxi)))
		f.write("\n")
		f.write("{},".format(len(cluster_lists[i])))
		f.write("{}".format(int(cluster_record_count[i])))
		f.write("\n")
		for j in range(0,10):
			f.write("{},".format(payment_norm_list[i][j]))
		f.write("\n")
		f.close()