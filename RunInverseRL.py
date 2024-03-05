import numpy as np
from loguru import logger

from sim4ad.opendrive import Map
from sim4ad.irlenv import IRLEnv
from sim4ad.data.data_loaders import DatasetDataLoader


class IRL:
    feature_num = 8
    n_iters = 200
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lam = 0.01
    lr = 0.05

    def __init__(self):
        self.buffer = []
        self.human_traj_features = []
        self.theta = None

        self.save_buffer = False

    @staticmethod
    def load_dataset():
        """Loading clustered trajectories"""
        data_loader = DatasetDataLoader(f"scenarios/configs/automatum.json")
        data_loader.load()

        episodes = data_loader.scenario.episodes

        return episodes

    def get_simulated_features(self):
        """get the features of forward simulations as well as human driver features"""
        # load the dataset
        episodes = self.load_dataset()

        for episode in episodes:
            # load the open drive map
            scenario_map = Map.parse_from_opendrive(episode.map_file)

            # for each agent
            for aid, agent in episode.agents.items():
                if aid != '7f8e80a7-da54-4783-bcfe-c4cb3d5e7067':
                    continue
                logger.info(f"Ego agent: {aid}")

                irl_env = IRLEnv(episode=episode, scenario_map=scenario_map, ego=agent, IDM=False)
                terminated = False
                for inx, t in enumerate(agent.time):
                    if t < 13.813813813813816:
                        continue
                    logger.info(f"Simulation time: {t}")

                    irl_env.reset(reset_time=t)

                    # set up buffer of the scene
                    buffer_scene = []

                    # only one point is alive, continue
                    if irl_env.interval[1] == irl_env.interval[0]:
                        continue
                    lateral_offsets, target_speeds = irl_env.sampling_space()
                    # for each lateral offset and target_speed combination
                    for lateral in lateral_offsets:
                        for target_speed in target_speeds:
                            action = (lateral, target_speed)
                            features, terminated, info = irl_env.step(action)

                            # get the features
                            traj_features = features[:-1]
                            human_likeness = features[-1]

                            # add scene trajectories to buffer
                            buffer_scene.append((lateral, target_speed, traj_features, human_likeness))

                            # set back to previous step
                            irl_env.reset(reset_time=t)

                    # calculate human trajectory feature
                    logger.info("Compute human driver features.")
                    irl_env.reset(reset_time=t, human=True)
                    features, terminated, info = irl_env.step()

                    if terminated or features[-1] > 2.5:
                        continue

                    # process data
                    human_traj = features[:-1]
                    buffer_scene.append([0, 0, features[:-1], features[-1]])

                    # add to buffer
                    if self.save_buffer:
                        self.human_traj_features.append(human_traj)
                        self.buffer.append(buffer_scene)

    def normalize_features(self):
        """normalize the features"""
        features = []
        for buffer_scene in self.buffer:
            for traj in buffer_scene:
                features.append(traj[2])
        max_v = np.max(features, axis=0)
        max_v[6] = 1.0

        for f in features:
            for i in range(IRL.feature_num):
                f[i] /= max_v[i]

    def maxent_irl(self):
        """training the weights using the buffer"""
        # initialize weights
        self.theta = np.random.normal(0, 0.05, size=IRL.feature_num)

        # iterations

        pm = None
        pv = None
        grad_log = []
        human_likeness_log = []

        for iteration in range(IRL.n_iters):
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

                # select trajectories tp calculate human likeness
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

            # update weights
            if pm is None:
                pm = np.zeros_like(grad)
                pv = np.zeros_like(grad)

            pm = IRL.beta1 * pm + (1 - IRL.beta1) * grad
            pv = IRL.beta2 * pv + (1 - IRL.beta2) * (grad * grad)
            mhat = pm / (1 - IRL.beta1 ** (iteration + 1))
            vhat = pv / (1 - IRL.beta2 ** (iteration + 1))
            update_vec = mhat / (np.sqrt(vhat) + IRL.eps)
            self.theta += IRL.lr * update_vec


def main():
    irl_instance = IRL()
    # compute features
    irl_instance.get_simulated_features()

    # normalize features
    # irl_instance.normalize_features()

    # Run MaxEnt IRL
    # irl_instance.maxent_irl()

    # Further steps for evaluation and testing of the learned model
    # ...
    pass


if __name__ == "__main__":
    main()
