import numpy as np
from loguru import logger
from multiprocessing import Pool
import pickle

from sim4ad.opendrive import Map
from sim4ad.irlenv import IRLEnv


class IRL:
    feature_num = 8
    n_iters = 200
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lam = 0.01
    lr = 0.05

    def __init__(self, episodes=None,
                 multiprocessing: bool = False,
                 num_processes: int = 12,
                 save_training_log: bool = False,
                 save_buffer: bool = False):
        self.buffer = []
        self.human_traj_features = []
        # initialize weights
        self.theta = np.random.normal(0, 0.05, size=IRL.feature_num)
        self.episode = None
        self.scenario_map = None
        self.pm = None
        self.pv = None

        self.training_log = {'iteration': [], 'average_feature_difference': [],
                             'average_log-likelihood': [],
                             'average_human_likeness': [],
                             'theta': []}

        self.save_buffer = save_buffer
        self.save_training_log = save_training_log

        self.multiprocessing = multiprocessing
        self.num_processes = num_processes
        self.episodes = episodes

    def get_feature_one_agent(self, item):
        """get the feature for one agent"""
        human_traj_features_one_agent = []
        buffer_one_agent = []
        aid, agent = item

        logger.info(f"Ego agent: {aid}")

        irl_env = IRLEnv(episode=self.episode, scenario_map=self.scenario_map, ego=agent, IDM=False)
        for inx, t in enumerate(agent.time):
            # if the agents is reaching its life end, continue, because the planned trajectory is not complete and few
            # points exist
            if agent.time[-1] - t < 1:
                continue
            logger.info(f"Simulation time: {t}")

            irl_env.reset(reset_time=t)

            # only one point is alive, continue
            if irl_env.interval[1] == irl_env.interval[0]:
                continue

            buffer_scene = irl_env.get_buffer_scene(t)

            # calculate human trajectory feature
            logger.info("Compute human driver features.")
            irl_env.reset(reset_time=t, human=True)
            features, terminated, info = irl_env.step()

            if terminated or features[-1] > 2.5:
                continue

            # process data
            human_traj = features[:-1]
            buffer_scene.append([0, 0, features[:-1], features[-1]])

            # save to buffer
            human_traj_features_one_agent.append(human_traj)
            buffer_one_agent.append(buffer_scene)

        return human_traj_features_one_agent, buffer_one_agent

    def get_simulated_features(self):
        """get the features of forward simulations as well as human driver features"""
        for episode in self.episodes:
            self.episode = episode
            # load the open drive map
            self.scenario_map = Map.parse_from_opendrive(episode.map_file)

            if self.multiprocessing:
                with Pool(processes=self.num_processes) as pool:
                    results = pool.map(self.get_feature_one_agent, episode.agents.items())
                if self.save_buffer:
                    for res in results:
                        if res is not None:
                            self.human_traj_features.extend(res[0])
                            self.buffer.extend(res[1])
            else:
                for aid, agent in episode.agents.items():
                    human_traj_features_one_agent, buffer_one_agent = self.get_feature_one_agent((aid, agent))
                    if self.save_buffer:
                        self.human_traj_features.extend(human_traj_features_one_agent)
                        self.buffer.extend(buffer_one_agent)

    def normalize_features(self):
        """normalize the features"""
        assert len(self.buffer) > 0, "Buffer is empty."

        features = []
        for buffer_scene in self.buffer:
            for traj in buffer_scene:
                features.append(traj[2])
        max_feature = np.max(features, axis=0)
        # set maximum collision value to 1 to avoid divided by zero
        max_feature[6] = 1.0
        # set social impact to 1 to avoid divided by zero
        if max_feature[7] == 0:
            max_feature[7] = 1.0

        for f in features:
            for i in range(IRL.feature_num):
                f[i] /= max_feature[i]

        # save max_v for normalize features during evaluation
        with open('max_feature.txt', 'w') as f:
            for item in max_feature:
                f.write("%s\n" % item)

        # save buffer data to avoid repeated computation
        if self.save_buffer:
            logger.info('Saved buffer data.')
            with open("buffer.pkl", "wb") as file:
                pickle.dump([self.human_traj_features, self.buffer], file)

    def maxent_irl(self, iteration):
        """training the weights under each iteration"""

        logger.info(f'interation: {iteration + 1}/{IRL.n_iters}')
        # fix collision feature's weight
        self.theta[6] = -10

        feature_exp = np.zeros([IRL.feature_num])
        human_feature_exp = np.zeros([IRL.feature_num])
        index = 0
        log_like_list = []
        iteration_human_likeness = []
        num_traj = 0

        for scene in self.buffer:
            # compute on each scene
            scene_trajs = []
            for trajectory in scene:
                reward = np.dot(trajectory[2], self.theta)
                scene_trajs.append((reward, trajectory[2], trajectory[3]))  # reward, feature vector, human likeness

            # calculate probability of each trajectory
            rewards = [traj[0] for traj in scene_trajs]
            probs = [np.exp(reward) for reward in rewards]
            probs = probs / np.sum(probs)

            # calculate feature expectation with respect to the weights
            traj_features = np.array([traj[1] for traj in scene_trajs])
            feature_exp += np.dot(probs, traj_features)  # feature expectation

            # calculate likelihood
            log_like = np.log(probs[-1] / np.sum(probs))
            log_like_list.append(log_like)

            # select trajectories to calculate human likeness
            # extracting the indices of the top 3 highest values in probs
            idx = probs.argsort()[-3:][::-1]
            iteration_human_likeness.append(np.min([scene_trajs[i][-1] for i in idx]))

            # calculate human trajectory feature
            human_feature_exp += self.human_traj_features[index]

            # go to next trajectory
            num_traj += 1
            index += 1

        # compute gradient
        grad = human_feature_exp - feature_exp - 2 * IRL.lam * self.theta
        grad = np.array(grad, dtype=float)

        # update weights using Adam optimization
        if self.pm is None:
            self.pm = np.zeros_like(grad)
            self.pv = np.zeros_like(grad)

        self.pm = IRL.beta1 * self.pm + (1 - IRL.beta1) * grad
        self.pv = IRL.beta2 * self.pv + (1 - IRL.beta2) * (grad * grad)
        mhat = self.pm / (1 - IRL.beta1 ** (iteration + 1))
        vhat = self.pv / (1 - IRL.beta2 ** (iteration + 1))
        update_vec = mhat / (np.sqrt(vhat) + IRL.eps)
        self.theta += IRL.lr * update_vec

        # record info during the training
        self.training_log['iteration'].append(iteration + 1)
        self.training_log['average_feature_difference'].append(
            np.linalg.norm(human_feature_exp / num_traj - feature_exp / num_traj))
        self.training_log['average_log-likelihood'].append(np.sum(log_like_list) / num_traj)
        self.training_log['average_human_likeness'].append(np.mean(iteration_human_likeness))
        self.training_log['theta'].append(self.theta)
