# Massimiliano Tamborski's dissertation code

### Requirements
We use the [AUTOMATUM](https://automatum-data.com/) dataset. Please first install the Python utility to interface the data
`pip3 install openautomatumdronedata`.

### Preprocessing
The preprocessing was done bt Dr. Cheng Wang. The code is available at `preprocessing.py`.

### The folder which I have created and I solely worked on are:
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