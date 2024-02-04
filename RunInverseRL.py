import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from sim4ad.opendrive import Map, plot_map
from sim4ad.irlenv import IRLEnv


# load clustered trajectories
def load_data():
    """Loading the demonstrations for training"""
    demonstrations = []
    data_path = 'scenarios/data/trainingdata'
    folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
    for folder in folders:
        folder_path = os.path.join(data_path, folder, 'irl.pkl')
        with open(folder_path, 'rb') as file:
            # Load the contents from the file
            data = pickle.load(file)
        demonstrations.append(data)

    return data


def compute_features(data):
    """Compute the features of each trajectory"""
    # ego motion

    # feature array
    features = np.array([ego_speed, abs(ego_longitudial_acc), abs(ego_lateral_acc), abs(ego_longitudial_jerk),
                         THWF, THWB, collision, social_impact, ego_likeness])

    return features


def reward_function(weights, features):
    return np.dot(weights, features)


def maxent_irl(feature_num: int, n_iters: int, scene_trajs, lam, lr):
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
        feature_exp += np.dot(probs, traj_features)  # feature expectation

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

    debug = True
    # load the static map
    scenario_map = Map.parse_from_opendrive(
        "scenarios/data/automatum/hw-a9-appershofen-001-d8087340-8287-46b6-9612-869b09e68448/staticWorld.xodr")

    # Extract demonstrations (trajectories) from the data
    agents = load_data()

    plot_map(scenario_map, markings=True, midline=False, drivable=True, plot_background=False)

    # for each agent
    for aid, agent in agents.items():
        sampled_trajectories = []

        # for each time step
        for inx, t in enumerate(agent.time):
            irl_agent = IRLEnv(agent=agent, current_inx=inx, scenario_map=scenario_map)

            lateral_offsets, target_speeds = irl_agent.sampling_space()
            # for each lateral offset and target_speed combination
            for lateral in lateral_offsets:
                for target_speed in target_speeds:
                    # 5 is the horizontal time
                    action = (lateral, target_speed, 5)
                    irl_agent.trajectory_planner(*action)
                    sampled_trajectories.append(irl_agent.planned_trajectory)

            # visualize the planned trajectories
            if debug:
                trajectories_local = []
                for trj in sampled_trajectories:
                    trj_local = []
                    for p in trj:
                        trj_local.append(irl_agent.position_local(s=p[0], d=p[1]))
                    trajectories_local.append(np.array(trj_local))
                for trj in trajectories_local:
                    plt.plot(trj[:, 0], trj[:, 1], linewidth=1)

                plt.show()

    # compute demonstration features
    demonstration_features = compute_features(agents)

    # Run MaxEnt IRL
    learned_reward_weights = maxent_irl()

    # Further steps for evaluation and testing of the learned model
    # ...


if __name__ == "__main__":
    main()
