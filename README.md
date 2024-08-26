# Sim4AD

### Requirements
We use the [AUTOMATUM](https://automatum-data.com/) dataset. Please first install the Python utility to interface the data
`pip3 install openautomatumdronedata`.

### Preprocessing
- Run `feature_normalization.py` for mean and standard deviation calculation of features in order to normalize features
- Run `preprocessing.py` for data generation for BC and offline RL. `feature_normalization.pkl` and `XXXtraining_log.pkl` 
are needed for preprocessing.
- 
## todo: need to preprocess all of the sriving style groups. You need to pass the --driving_style_idx argument from 0 to 2

### Evaluation folders:
- /baselines: contains the code for the baselines
- /evaluation: contains the  code for the metrics (and their unit tests), scoring, evaluation and plotting.
- /simulator: which contains the lightweigth simulator and the RL simulator


### Useful commands to run the code are the following:
- `python3 evaluation/evaluation_methods.py` to evaluate the baselines 
- `baselines/bc_baseline.py` to train the BC baseline
- `baselines/rl.py` to train the SAC baseline
- `simulator/lightweight_simulator.py` to run the lightweight simulator. Useful parameters to 
set (at the bottom of the file) are `spawn_method` = which is "dataset_one" for micro analysis 
and "dataset_all" for macro analysis, `policy_type` = which is "bc-all-obs-5_pi" for BC, "sac_5_rl" for SAC (after training it as above),
or "idm" for IDM

- pip install -e . in the root folder of sim4ad # todo update the requirements
t
- install Pytorch from the official website
- `pip install -r requirements.txt` in the root folder of sim4ad

- maps: change line `<OpenDRIVE xmlns="http://www.opendrive.org">` to `<OpenDRIVE>` in each of the map if there is an error about a None header not having attribute `name`
