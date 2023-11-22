from util import parse_args
import sys
import numpy as np
import matplotlib.pyplot as plt

from sim4ad.data.data_loaders import DatasetDataLoader


class FeatureExtraction:

    @staticmethod
    def get_root_mean_square(data):
        N = len(data)
        # Root-mean-square
        data_rms = np.sqrt(sum([data[i] ** 2 for i in range(N)]) / N)

        return data_rms

    @staticmethod
    def get_average_rectified_value(data):
        N = len(data)
        # Average rectified value
        lat_acc_arv = sum([abs(data[i]) for i in range(N)]) / N

        return lat_acc_arv

    def get_shape_factor(self, data):
        # Shape factor of lateral acceleration
        # Root-mean-square
        data_rms = self.get_root_mean_square(data)
        # Average rectified value
        data_arv = self.get_average_rectified_value(data)

        # Shape factor
        data_sf = data_rms / data_arv

        return data_sf

    @staticmethod
    def get_standard_devisation(data):
        return np.std(data)

    @staticmethod
    def get_frequency_centroid(data):
        N = len(data)
        dft = np.fft.fft(data)
        magnitude = np.abs(dft)
        freq = np.fft.fftfreq(N)
        freq_centroid = sum([freq[i] * magnitude[i] for i in range(N)]) / N

        return freq_centroid


def plot(freq, magnitude):
    # plt.figure()
    # plt.scatter(t, data, color='r')

    plt.figure()
    plt.stem(freq, magnitude)
    plt.show()


def main():
    args = parse_args()
    data_loader = DatasetDataLoader(f"scenarios/configs/{args.map}.json")
    data_loader.load()

    feature_extractor = FeatureExtraction()

    for episode in data_loader.scenario.episodes:
        features = {}
        for agent_id, agent in episode.agents.items():

            long_acc = agent.ax_vec
            lat_acc = agent.ay_vec
            velocity = [np.sqrt(agent.vx_vec[i] ** 2 + agent.vy_vec[i] ** 2) for i in range(len(agent.vx_vec))]

            long_acc_fc = feature_extractor.get_frequency_centroid(long_acc)
            lat_acc_sf = feature_extractor.get_shape_factor(lat_acc)
            lat_acc_rms = feature_extractor.get_root_mean_square(lat_acc)
            velocity_std = feature_extractor.get_standard_devisation(velocity)
            features[agent_id] = [long_acc, long_acc_fc, lat_acc_sf, lat_acc_rms, velocity_std]


            print('ok')


if __name__ == '__main__':
    sys.exit(main())
