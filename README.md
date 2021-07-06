TaxiTrip-Synthesizer
==
***Team GooseDP Solution to Differential Privacy Temporal Map Challenge (DeID2)-Sprint 3***

-----

## Brief Introduction
We are team GooseDP from the University of Waterloo. We finished the 5th in the [NIST Temporal Map Challenge: **Sprint 3**](https://www.drivendata.co/blog/differential-privacy-winners-sprint3/). This repository is our open-sourced solution to the challenge. The diagram below illustrates a summary of our approach and for full details we refer to the technical report in this repository ([`NIST_DP_Privacy_GooseDP_Writeup.pdf`](https://github.com/ctcovington/goosedp_sprint3_open_source/blob/main/NIST_DP_Privacy_GooseDP_Writeup.pdf)). If anyone wants to generalize this approach to datasets in other domains, we have some suggested guidelines located here ([`Approach_to_Generalization.pdf`](https://github.com/ctcovington/goosedp_sprint3_open_source/blob/main/Approach_to_Generalization.pdf)).
![Overall_Approach](./Overall_Approach.png)

## Submission Repository Structure
    ├── Submission directory/
    │   ├── Step0_Archetype_Generation/             *Step-0: Preprocessing
    |       ├── Results_GMM/
    |       └── k_archetypes.py 
    │   ├── Step1_Archetype_Counting/               *Step-1: Private Analysis
    |       └── archetype_company_counts.py    
    │   ├── Step2_ Synthetic_Data_Generation/       *Step-2: Synthetic Record Generation
    |       ├── sample_triplets.py
    |       └── post_col_generation.py
    |   ├── data/                                   *Ground Truth Data and Parameters File
    |       ├── parameters.json
    |       ├── (public_data.csv)                   *Public Dataset
    |       └── (ground_truth.csv)                  *Private Dataset
    |   ├── main.py                                 *Program Entrance
    |   ├── requirements.txt                        *Package Requirements
    |   ├── NIST_DP_Privacy_GooseDP_Writeup.pdf     *Technical Report
    |   └── Approach_to_Generalization.pdf          *Generalization Guidance

## Execution Commands
If you want to run our submission manually, first put the private dataset (`ground_truth.csv` file) and the public dataset (`public_data.csv` file) under the `data/`directory, and install the required packages.

>`pip install -r requirements.txt`

Then run the command to execute the main file.

>`python main.py`

## Code Guide
**Main Function**  (`main.py`) <br>
The program entrance to our code submission. <br>
*We create a script `create_submission.sh` to help zip our submission code files.* <br><br>
**Step 0: Preprocessing** (`Step0_Archetype_Generation/`) <br>
The preprocessing step in the write-up is corresponding to the contents in the `Archetype_Generation/`directory.  <br>
Under this directory, the file `k_archetypes.py` is used for archetype generation and the generated archetype information files are stored in the `Results_GMM/` directory. <br>
*Note:* This step only uses the public dataset, therefore we create the archetype files locally and associate those files in the submission. <br><br>
**Step 1: Private Analysis** (`Step1_Archetype_Counting/`) <br>
The private analysis step in the write-up is corresponding to the contents in the `Archetype_Counting/`directory. <br>
Under this directory, the file `archetype_company_counts.py` is used for creating private histograms over the private dataset (details referring to the write-up) and returning privatized counts of taxis and companies. <br><br>

**Step 2:  Synthetic Data Generation**  (`Step2_sample_triplets/`)  <br>

**Synthesize Taxi-trips Record** (`sample_triplets.py`) <br>
The synthetic record step in the write-up is corresponding to the contents in the `Step2_sample_triplets/`directory. <br>
Under this directory, the file `sample_triplets.py` is used for generating synthetic records for `('taxi_id', 'shift', 'company_id', 'pickup_community_area', 'dropoff_community_area')` columns. <br>

**Synthesize Other Columns**  (`post_col_generation.py`) <br>
The post processing step in the write-up is corresponding to the contents in the `Step3_nonprivate_gen/`directory. <br>
Under this directory, the file `post_col_generation.py` is used for generating synthetic records for the rest of the columns, i.e., `('fare', 'trip_miles', 'trip_seconds', 'tips', 'trip_total', 'payment_type')`, based on the k-marginals. <br>

## How to Cite: 

> ```
> @misc{GooseDP_Syn,
>   author = {Covington, Christian and Knopf, Karl and Mohapatra, Shubhankar and Zhang, Shufan},
>   title = {TaxiTrip-Synthesizer: Team GooseDP Solution to Differential Privacy Temporal Map Challenge (DeID2)-Sprint 3},
>   year = {2021},
>   publisher = {GitHub},
>   journal = {GitHub repository},
>   howpublished = {\url{https://github.com/ctcovington/goosedp_sprint3_open_source}}
> }
> ```


## Team Members: 
[Christian Covington](mailto:ccovington@uwaterloo.ca) <br>
[Karl Knopf](mailto:kknopf@uwaterloo.ca)  <br>[Shubhankar Mohapatra](mailto:shubhankar.mohapatra@uwaterloo.ca) <br>[Shufan Zhang](mailto:shufan.zhang@uwaterloo.ca) <br>

