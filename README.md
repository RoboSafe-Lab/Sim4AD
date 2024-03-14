# Sim4AD
RL/MARL-based Behavior Modeling for Simulation-Driven Autonomous Vehicle Validation


### Requirements
We use the [AUTOMATUM](https://automatum-data.com/) dataset. Please first install the Python utility to interface the data
`pip3 install openautomatumdronedata`.

### Clustering
We cluster the driving styles and train different policies for each cluster.
we have two different methods for clustering.

Method 1: using features defined from this paper. 
`run clustering.py` to have the results. 

Method 2: we segment each trajectory into low risk, safe, potential danger and high risk, 
which are then used as features for clustering. The risk level is determined by inverse TTC and THW.
`run riskclustering.py` to show the radar charts, which indicates different clusters.

### IRL
After training, the `training_log.pkl` records important info during training.
`run irl_training_analysis.py` to visualize the change of variables during training.

`run evaluation.py` under *sim4ad/irlenv* to test the trained reward function.