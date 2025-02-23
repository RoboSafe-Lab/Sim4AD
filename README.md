# Sim4AD

### Requirements
We use the [AUTOMATUM](https://automatum-data.com/) dataset. Please first install the Python utility to interface the data
`pip3 install openautomatumdronedata`.

### Preprocessing
- Run `feature_normalization.py` for mean and standard deviation calculation of features in order to normalize features
- Run `feature_extraction_irl.py` to extract and compute the feature values for the dataset
- Run `preprocessing.py` for data generation for BC and offline RL. `feature_normalization.pkl` and `XXXtraining_log.pkl` 
are needed for preprocessing.
- 

### IRL to infer reward functions
- Run `training_irl.py` to train the feature weights for each driving style and construct a separate linear reward function for each style.

### Train policies for every driving style
- Run `baselines/bc_baseline.py` to train the BC baseline
- Run `sim4ad/offlinerlenv/td3bc_automatum.py to train offline RL
- Run `baselines/sac/model.py to train the SAC baseline
- Run `baselines/sac/train_multicar.py to train the MARL policies

### Evaluation folders:
- /baselines: contains the code for the baselines
- /evaluation: contains the  code for the metrics (and their unit tests), scoring, evaluation and plotting.
- /simulator: which contains the lightweigth simulator and the RL simulator


### Other commands to run the code are the following:
- `demonstrtation_analysis.py`,'reward_analysis.py" to draw the reward distribution 
- `run_evaluations.py` to deploy the policy and visualize the effects of the training. The script supports two modes:'dataset one' and 'dataset all'. 
- `irl_training_analysis.py` to visualize the reward function weights obtained from IRL training.
- `simulator/lightweight_simulator.py` to run the lightweight simulator. Useful parameters to 
set (at the bottom of the file) are `spawn_method` = which is "dataset_one" for micro analysis 
and "dataset_all" for macro analysis, `policy_type` = which is "bc-all-obs-5_pi" for BC, "sac_5_rl" for SAC (after training it as above),
or "idm" for IDM

- pip install -e . in the root folder of sim4ad # todo update the requirements
t
- install Pytorch from the official website
- `pip install -r requirements.txt` in the root folder of sim4ad

- maps: change line `<OpenDRIVE xmlns="http://www.opendrive.org">` to `<OpenDRIVE>` in each of the map if there is an error about a None header not having attribute `name`
