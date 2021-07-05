import json
from pathlib import Path

import numpy as np
import pandas as pd
import Step1_Archetype_Counting.analytic_gaussian as ag
from Step1_Archetype_Counting.archetype_company_counts import get_private_counts
from Step2_ Synthetic_Data_Generation.sample_triplets import sample_triplets
from Step2_ Synthetic_Data_Generation.post_col_generation import post_col_generation

ROOT_DIRECTORY = Path("./")
RUNTIME_DIRECTORY = ROOT_DIRECTORY / "submission"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"

DEFAULT_GROUND_TRUTH = DATA_DIRECTORY / "ground_truth.csv"
DEFAULT_ARCHETYPES = ROOT_DIRECTORY / "Step0_Archetype_Generation" / "Results_GMM" / "Results_10"
DEFAULT_PRIVATE_COUNTS = ROOT_DIRECTORY / "Step1_Archetype_Counting"
DEFAULT_TRIPLETS = ROOT_DIRECTORY / "Step2_ Synthetic_Data_Generation"
DEFAULT_SAMPLING = ROOT_DIRECTORY / "Step2_ Synthetic_Data_Generation"
DEFAULT_PUBLIC_DATA = DATA_DIRECTORY / "public_data.csv"
DEFAULT_PARAMS = DATA_DIRECTORY / "parameters.json"
DEFAULT_OUTPUT = ROOT_DIRECTORY / "submission.csv"
ARCHETYPE_BUDGET_PROP = 0.1

NOT_USED_IN_SUBMISSION = ("trip_hour_of_day", "trip_day_of_week")

def main(parameters_file: Path = DEFAULT_PARAMS,
    ground_truth_file: Path = DEFAULT_GROUND_TRUTH,
	archetype_dir: Path = DEFAULT_ARCHETYPES,
	private_counts_dir: Path = DEFAULT_PRIVATE_COUNTS,
	triplets_dir: Path = DEFAULT_TRIPLETS,
	sampling_dir: Path = DEFAULT_SAMPLING,
	public_data_file: Path = DEFAULT_PUBLIC_DATA,
    output_file: Path = DEFAULT_OUTPUT,
	archetype_budget_prop = ARCHETYPE_BUDGET_PROP):

	'''
	load parameter information
	'''
	with parameters_file.open("r") as fp:
		parameters = json.load(fp)

	epsilons = [run["epsilon"] for run in parameters["runs"]]
	deltas = [run["delta"] for run in parameters["runs"]]

	columns = [k for k in parameters["schema"].keys() if k not in NOT_USED_IN_SUBMISSION]
	headers = ["epsilon"] + columns

	dtypes = {column_name: d["dtype"] for column_name, d in parameters["schema"].items()}
	dtypes.pop('trip_day_of_week')
	dtypes.pop('trip_hour_of_day')

	'''
	construct synthetic data
	'''
	out = pd.DataFrame(columns=headers)
	for epsilon, delta in zip(epsilons, deltas):
		# get private archetype and company_id counts 
		get_private_counts(ground_truth_file, parameters_file, archetype_dir, private_counts_dir, epsilon, delta, archetype_budget_prop)

		# generate shift/pickup/dropoff and company_id from private counts 
		sample_triplets(parameters_file, archetype_dir, private_counts_dir, triplets_dir, epsilon)

		# generate other columns 
		post_col_generation(public_data_file, triplets_dir, sampling_dir, epsilon)

		# create final data
		final = pd.read_csv(str(sampling_dir)+f'/final_dataset_{epsilon}.csv')
		final = final[columns]
		final['epsilon'] = epsilon
		final = final[headers]

		out = out.append(final)
		print(f'finished synthesizing for epsilon{epsilon}')

	# enforce data bounds 
	for col, entry in parameters["schema"].items():
		if 'min' in entry:
			out[col] = np.clip(out[col], entry['min'], np.inf)
		if 'max' in entry:
			out[col] = np.clip(out[col], -np.inf, entry['max'])

	# convert column types
	for col, col_type in dtypes.items():
		out[col] = out[col].astype(col_type)

	out.to_csv(output_file, index=False)

if __name__ == '__main__':
    main()