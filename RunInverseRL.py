import numpy as np
from loguru import logger
import matplotlib.pyplot as plt

from sim4ad.opendrive import Map
from sim4ad.irlenv import IRLEnv, utils
from sim4ad.data.data_loaders import DatasetDataLoader


# load clustered trajectories
def load_dataset():
    """Loading the dataset"""
    data_loader = DatasetDataLoader(f"scenarios/configs/automatum.json")
    data_loader.load()

    episodes = data_loader.scenario.episodes

    return episodes


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

    debug = False

    # load the dataset
    episodes = load_dataset()

    for episode in episodes:
        # load the opendrive map
        scenario_map = Map.parse_from_opendrive(episode.map_file)

        # for each agent
        for aid, agent in episode.agents.items():
            sampled_trajectories = []
            # if aid != '7f8e80a7-da54-4783-bcfe-c4cb3d5e7067':
            #     continue
            logger.info(f"Ego agent: {aid}")

            irl_env = IRLEnv(episode=episode, scenario_map=scenario_map, ego=agent, IDM=False)
            terminated = False
            for inx, t in enumerate(agent.time):
                # if t < 11.077744411077745:
                #     continue
                logger.info(f"Simulation time: {t}")

                irl_env.reset(reset_time=t)

                # calculate human trajectory feature
                # features, terminated, info = irl_env.step()

                # only one point is alive, continue
                if irl_env.interval[1] == irl_env.interval[0]:
                    continue
                lateral_offsets, target_speeds = irl_env.sampling_space()
                # for each lateral offset and target_speed combination
                for lateral in lateral_offsets:
                    for target_speed in target_speeds:
                        action = (lateral, target_speed)
                        features, terminated, info = irl_env.step(action)

                        sampled_trajectories.append(irl_env.vehicle.planned_trajectory)

                        # set back to previous step
                        irl_env.reset(reset_time=t)

                if terminated:
                    continue

                # visualize the planned trajectories
                if debug:
                    trajectories_local = []
                    for trj in sampled_trajectories:
                        trj_local = []
                        for p in trj:
                            trj_local.append(
                                utils.frenet2local(reference_lane=irl_env.vehicle.lane, s=p[0], d=p[1]))
                        trajectories_local.append(np.array(trj_local))
                    for trj in trajectories_local:
                        plt.plot(trj[:, 0], trj[:, 1], linewidth=1)

                    plt.show()

    # compute demonstration features
    # demonstration_features = compute_features(agents)

    # Run MaxEnt IRL
    # learned_reward_weights = maxent_irl()

    # Further steps for evaluation and testing of the learned model
    # ...
    pass


if __name__ == "__main__":
    main()
