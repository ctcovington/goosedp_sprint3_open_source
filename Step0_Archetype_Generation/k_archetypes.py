import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd 
#import csv
from collections import defaultdict
import pickle
import os 

num_clusters = 13
charts_on = False
res_dir = os.path.join('CC_Results_GMM', 'Results_{0}'.format(num_clusters))
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# Read in datafile
#data = []
#with open("ground_truth.csv",'r') as csvfile:
#	title = csvfile.readline()
#	reader = csv.reader(csvfile,quoting = csv.QUOTE_NONNUMERIC)
#	for row in reader:
#		data.append(row)

#data = np.array([np.array(xi) for xi in data])
data = pd.read_csv("ground_truth.csv")
# data = df.to_numpy() #np.array([np.array(xi) for xi in data])


#seperate out the triplets (pickup_area,dropoff_area,shift) and scale them

# features = np.column_stack((data[:,1],data[:,3],data[:,4]))
# # features = np.column_stack((data[:,1],data[:,3]))
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)

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

#Determine k clusters of the triplets using the k-means algorithm
#kmeans = KMeans(
#    init="random",
#    n_clusters= num_clusters,
#    n_init=10,
#    max_iter=300,
#    random_state=42 # randomization seed value
#)

gmm = GaussianMixture(n_components=num_clusters)
gmm.fit(counts)

#kmeans.fit(counts)

labels = gmm.predict(counts)
#labels = kmeans.labels_
pkl_filename = os.path.join(res_dir, "archetypes.pkl")
with open(pkl_filename, 'wb') as file:
    pickle.dump(gmm, file)
data = data.to_numpy()
# The lowest SSE value
#print(kmeans.inertia_)
# Final locations of the centroid
#print(kmeans.cluster_centers_)
# The number of iterations required to converge
#print(kmeans.n_iter_)
#kmeans.labels_
#print(kmeans.labels_)

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

for i in range(0,num_clusters):
	print("Archetype {}:".format(i))
	print("Size = {}".format(len(cluster_lists[i])))
	print("Proportion = {}".format(len(cluster_lists[i]) / taxi_count))

# Determine the Distributions of the Triplets under this archetype

pickup_dists = [ defaultdict(int) for _ in range(num_clusters)]
dropoff_dists = [ defaultdict(int) for _ in range(num_clusters)]
shift_dists = [ defaultdict(int) for _ in range(num_clusters)]
payment_dists =  [ defaultdict(int) for _ in range(num_clusters)]

taxi_id = data[0][0]
current_cluster = 0
for i in range(num_clusters):
	if data[0][0] in cluster_lists[i]:
		current_cluster = i
		break

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

for i in range(0,num_clusters):
	print("Archetype {}:".format(i))
	print("Shift Dist: {}".format(shift_dists[i]))
	print("Pickup Dist: {}".format(pickup_dists[i]))
	print("Dropoff Dist: {}".format(dropoff_dists[i]))
	print("Payment Dist: {}".format(payment_dists[i]))
			

# Plot Distributions

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


if charts_on == True:
	for i in range(num_clusters):
		pickup_plot, ax1 = plt.subplots()
		pickup_plots = ax1.bar(list(pickup_normalized[i].keys()), pickup_normalized[i].values(), color='r')
		pickup_plot_string = "pickup_{}.png".format(i)
		plt.savefig(pickup_plot_string)
		plt.close(pickup_plot)

		dropoff_plot, ax2 = plt.subplots()
		dropoff_plots = ax2.bar(list(dropoff_normalized[i].keys()), dropoff_normalized[i].values(), color='r')
		dropoff_plot_string = "dropoff_{}.png".format(i)
		plt.savefig(dropoff_plot_string)
		plt.close(dropoff_plot)

		shift_plot, ax3 = plt.subplots()	
		shift_plots = ax3.bar(list(shift_normalized[i].keys()), shift_normalized[i].values(), color='r')
		shift_plot_string = "shift_{}.png".format(i)
		plt.savefig(shift_plot_string)
		plt.close(shift_plot)


# Accuracy of Clusters on HOC Metric
shifts = defaultdict(int)
pickups = defaultdict(int)
dropoffs = defaultdict(int)

pickup_max = [ defaultdict(int) for _ in range(num_clusters)]
dropoff_max = [ defaultdict(int) for _ in range(num_clusters)]
shift_max = [ defaultdict(int) for _ in range(num_clusters)]


success = np.zeros(num_clusters)
taxi_id = data[0][0]
for i in range(num_clusters):
        if row[0] in cluster_lists[i]:
                current_cluster = i
                break
count = 0
failure = 0
cluster_record_count = np.zeros(num_clusters)
taxi_column = [] 
for row in data:
        if row[0] != taxi_id and count > 0:
                for i in range(num_clusters):
                        if taxi_id in cluster_lists[i]:	
                                for k in shifts:
                                        if abs(shifts[k] - shift_normalized[i][k]) > 5:
                                                failure += 1
                                        if abs(shifts[k] - shift_normalized[i][k]) > shift_max[i][k]:
                                                shift_max[i][k] = abs(shifts[k] - shift_normalized[i][k])
                                for k in pickups:
                                        if abs(pickups[k] - pickup_normalized[i][k]) > 5:
                                                failure += 1
                                        if abs(pickups[k] - pickup_normalized[i][k]) > pickup_max[i][k]:
                                                pickup_max[i][k] = abs(pickups[k] - pickup_normalized[i][k])
                                for k in dropoffs:
                                        if abs(dropoffs[k] - dropoff_normalized[i][k]) > 5:
                                                failure += 1
                                        if abs(dropoffs[k] - dropoff_normalized[i][k]) > dropoff_max[i][k]:
                                                dropoff_max[i][k] = abs(dropoffs[k] - dropoff_normalized[i][k])
                                if (failure / (21+78+78)) == 0.00: # all values are with 5 of the archetype distribution value
                                        success[i] += 1
                                break
                shifts = defaultdict(int)
                pickups = defaultdict(int)
                dropoffs = defaultdict(int)
                count = 0
                failure = 0
                taxi_id = row[0]
                for i in range(num_clusters):
                        if row[0] in cluster_lists[i]:
                                current_cluster = i
                                break
                        
        #otherwise store distribution information
                        
        pickups[row[3]] +=1
        dropoffs[row[4]] +=1
        shifts[row[1]] += 1
        count +=1
        cluster_record_count[current_cluster] += 1
        taxi_column.append(current_cluster)

# For last taxi_id
for i in range(num_clusters):
	if taxi_id in cluster_lists[i]:	
		for k in shifts:
			if abs(shifts[k] - shift_normalized[i][k]) > 5:
				failure += 1
			if abs(shifts[k] - shift_normalized[i][k]) > shift_max[i][k]:
				shift_max[i][k] = abs(shifts[k] - shift_normalized[i][k])
		for k in pickups:
			if abs(pickups[k] - pickup_normalized[i][k]) > 5:
				failure += 1
			if abs(pickups[k] - pickup_normalized[i][k]) > pickup_max[i][k]:
				pickup_max[i][k] = abs(pickups[k] - pickup_normalized[i][k])
		for k in dropoffs:
			if abs(dropoffs[k] - dropoff_normalized[i][k]) > 5:
				failure += 1
			if abs(dropoffs[k] - dropoff_normalized[i][k]) > dropoff_max[i][k]:
				dropoff_max[i][k] = abs(dropoffs[k] - dropoff_normalized[i][k])
		if (failure / (21+78+78)) == 0.00: # all values are with 5 of the archetype distribution value
			success[i] += 1
		break



for i in range(0,num_clusters):
	print("Success Rate of Archetype {}:\n".format(i))
	print(success[i]/len(cluster_lists[i]))
	print("Max Diff of Shift for Archetype {}:\n".format(i))
	print(shift_max[i])
	print("Max Diff of Pickup for Archetype {}:\n".format(i))
	print(pickup_max[i])
	print("Max Diff of Dropoff for Archetype {}:\n".format(i))
	print(dropoff_max[i])

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
# Determine 3-way marginal counts per archetype
#marginal_counts = df.groupby(['archetype','shift','pickup_community_area','dropoff_community_area'], as_index=False).mean('archetype')
#archetype_groups = marginal_counts.groupby('archetype').mean().to_frame()
#df.groupby(['archetype','shift','pickup_community_area','dropoff_community_area']).size()
#archetype_groups = df.groupby(['archetype'])
# Detemine 3-way marginal distributions per archetype
#marginal_counts = archetype_groups.value_counts(subset=['shift','pickup_community_area','dropoff_community_area'], normalize = True)
archetype_probs = df.groupby('archetype').size().div(len(df))

within_archetype_3way_marginals = df.groupby(['archetype', 'shift', 'pickup_community_area', 'dropoff_community_area']).size().div(len(df)).div(archetype_probs, axis=0, level='archetype').reset_index()

# Print marginal distributions to a new file
within_archetype_3way_marginals.to_csv(os.path.join(res_dir, 'archetype_marginals.csv'))


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
