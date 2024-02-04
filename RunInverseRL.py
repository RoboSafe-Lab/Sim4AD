import numpy as np
import pandas as pd
import os
import pickle


# load clustered trajectories
def load_demonstrations():
    """Loading the demonstrations for training"""
    demonstrations = []
    data_path = 'scenarios/data/trainingdata'
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    for folder in folders:
        with open(folder, 'rb') as file:
            # Load the contents from the file
            data = pickle.load(file)
        demonstrations.append(data)

    return data


def compute_features(data):
    """Compuste the features of each trajectory"""
    features = data
    return features


def reward_function(weights, features):
    return np.dot(weights, features)


def simulate_trj(vehicles):
    """sample trajectory for each vehicle at each timestep """
    for vech in vehicles:
        # run until the road ends
        for start in train_steps:
            # target sampling space
            lateral_offsets, target_speeds = env.sampling_space()

            # for each lateral offset and target_speed combination
            for lateral in lateral_offsets:
                for target_speed in target_speeds:
                    pass

        # calculate human trajectory feature at each step
        pass


def maxent_irl():
    # initialize weights
    theta = np.random.normal(0, 0.05, size=feature_num)

    # iterations
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    pm = None
    pv = None
    grad_log = []
    human_likeness_log = []

    for iteration in range(n_iters):
        feature_exp = np.zeros([feature_num])
        human_feature_exp = np.zeros([feature_num])

        # calculate probability of each trajectory
        rewards = [traj[0] for traj in scene_trajs]
        probs = [np.exp(reward) for reward in rewards]
        probs = probs / np.sum(probs)

        # calculate feature expectation with respect to the weights
        traj_features = np.array([traj[1] for traj in scene_trajs])
        feature_exp += np.dot(probs, traj_features) # feature expectation

        # compute gradient
        grad = human_feature_exp - feature_exp - 2 * lam * theta
        grad = np.array(grad, dtype=float)

        # update weights
        if pm is None:
            pm = np.zeros_like(grad)
            pv = np.zeros_like(grad)

        pm = beta1 * pm + (1 - beta1) * grad
        pv = beta2 * pv + (1 - beta2) * (grad * grad)
        mhat = pm / (1 - beta1 ** (iteration + 1))
        vhat = pv / (1 - beta2 ** (iteration + 1))
        update_vec = mhat / (np.sqrt(vhat) + eps)
        theta += lr * update_vec


def main():
    # parameters
    n_iters = 200
    lr = 0.05
    lam = 0.01
    feature_num = 8
    period = 0

    # Extract demonstrations (trajectories) from the data
    demonstrations = load_demonstrations()

    # Run MaxEnt IRL
    learned_reward_weights = maxent_irl()

    # Further steps for evaluation and testing of the learned model
    # ...


if __name__ == "__main__":
    main()
