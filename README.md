# HAD-Gen: Human-like and Diverse scenario generation 
This repository is the PyTorch implementation of our paper titled "HAD-Gen: Human-like and Diverse Agent Behavior Modeling for Controllable Scenario Generation".
## Results
* Our method HAD-Gen can controllably generate human-like and diverse scenarios and can be used for AVs testing and validation.
<div style="display: flex;">
  <img src="https://github.com/RoboSafe-Lab/Sim4AD/raw/Marl4IRL/images/Diversity.png" width="48%" />
  <img src="https://github.com/RoboSafe-Lab/Sim4AD/raw/Marl4IRL/images/Human-likeness.png" width="48%" />
</div>
* Compared with baselines, the policies we trained has the strongest generalization ability and can adapt to new real-world scenarios.
<p align="center">
  <img src="https://github.com/RoboSafe-Lab/Sim4AD/raw/Marl4IRL/images/Generalibility.png" width="80%" />
</p>


## Method
### The HAD-Gen framework
<p align="center">
  <img src="https://github.com/RoboSafe-Lab/Sim4AD/raw/Marl4IRL/images/framework.png" width="80%" />
</p>
    The overall working flow of our proposed method. The driving behavior in a dataset is clustered into aggressive, normal and cautious. Each
sub-dataset is then used to reconstruct a reward function representing human driving. The reconstructed rewards are fundamental for offline RL and MARL,
which generate a driving policy for each cluster. The various driving policies are deployed in a simulation given a policy selection strategy.

### Requirements
We use the [AUTOMATUM](https://automatum-data.com/) dataset. Please first install the Python utility to interface the data
`pip3 install openautomatumdronedata`.

### Run
- Run `riskclustering.py` to get the clustering results. This will save clustering results in a json file under scenarios/configs.
- Run `feature_extraction_irl.py` to extract feature values from the dataset, which will be subsequently used for irl. 
- Run `training_irl.py` to get MaxEnt IRL weights, which will be saved as xxxtraining_log.pkl.

- Run `feature_normalization.py` for mean and standard deviation calculation of features in order to normalize features
- Run `preprocessing.py` for data generation for BC and offline RL. `feature_normalization.pkl`(from feature_normalization.py) and `XXXtraining_log.pkl`(from training_irl.py) 
are needed for preprocessing.

- Run `td3bc_automatum.py` to get offline rl results
- Run `sac/model.py` to get log-replay results
- Run `train_multicar.py` to get online rl results (self-replay)
  
### Evaluation folders:
- /baselines: contains the code for the baselines
- /evaluation: contains the  code for the metrics (and their unit tests), scoring, evaluation and plotting.
- /simulator: which contains the lightweigth simulator and the RL simulator


### Useful commands to run the code are the following:
- `python3 evaluation/evaluation_methods.py` to evaluate the baselines 
- `baselines/bc_baseline.py` to train the BC baseline
- `baselines/model.py` to train the SAC baseline
- `simulator/lightweight_simulator.py` to run the lightweight simulator. Useful parameters to 
set (at the bottom of the file) are `spawn_method` = which is "dataset_one" for micro analysis 
and "dataset_all" for macro analysis, `policy_type` = which is "bc-all-obs-5_pi" for BC, "sac_5_rl" for SAC (after training it as above),
or "idm" for IDM

- pip install -e . in the root folder of sim4ad # todo update the requirements
t
- install Pytorch from the official website
- `pip install -r requirements.txt` in the root folder of sim4ad

- maps: change line `<OpenDRIVE xmlns="http://www.opendrive.org">` to `<OpenDRIVE>` in each of the map if there is an error about a None header not having attribute `name`

### Reference

@article{wang2025had,
  title={HAD-Gen: Human-like and Diverse Driving Behavior Modeling for Controllable Scenario Generation},
  author={Wang, Cheng and Kong, Lingxin and Tamborski, Massimiliano and Albrecht, Stefano V},
  journal={arXiv preprint arXiv:2503.15049},
  year={2025}
}