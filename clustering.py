from util import parse_args
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.mixture import GaussianMixture

from sim4ad.data.data_loaders import DatasetDataLoader
from visualization import Visualization


class FeatureExtraction:
    """
    features determined by the paper "Feature selection for driving style and skill clustering using naturalistic
    driving data and driving behavior questionnaire"
    """

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

    @staticmethod
    def get_max_value(data):
        return np.max(data)


class Clustering:
    """
    Various clustering methods can be defined here
    """
    @staticmethod
    def kmeans(dataframe):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(dataframe.iloc[:, 2:-1])
        # replace 3 with your chosen number of clusters
        kmeans = KMeans(init="random", n_init=10, n_clusters=3, max_iter=500)
        kmeans.fit(scaled_features)
        dataframe['cluster'] = kmeans.labels_
        clustered_dataframe = dataframe.groupby('cluster')

        return clustered_dataframe

    @staticmethod
    def hierarchical(dataframe):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(dataframe.iloc[:, 2:-1])

        # Hierarchical clustering
        hc = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
        y_hc = hc.fit_predict(scaled_features)

        # Plotting dendrogram
        linked = linkage(scaled_features, method='ward')
        plt.figure(figsize=(10, 7))
        dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        plt.show()

        return y_hc

    @staticmethod
    def GMM(dataframe):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(dataframe.iloc[:, 2:-1])

        # Create a Gaussian Mixture Model
        gmm = GaussianMixture(n_components=3, random_state=0)
        # Fit the model
        gmm.fit(scaled_features)
        # Predict the cluster for each data point
        dataframe['cluster'] = gmm.predict(scaled_features)
        clustered_dataframe = dataframe.groupby('cluster')

        return clustered_dataframe


def main():
    args = parse_args()
    data_loader = DatasetDataLoader(f"scenarios/configs/{args.map}.json")
    data_loader.load()

    feature_extractor = FeatureExtraction()
    features = {'episode': [], 'id': [], 'long_acc_max': [], 'long_acc_fc': [], 'lat_acc_sf': [],
                'lat_acc_rms': [], 'vel_std': [], 'cluster': None}

    for inx, episode in enumerate(data_loader.scenario.episodes):
        for agent_id, agent in episode.agents.items():
            long_acc = agent.ax_vec
            lat_acc = agent.ay_vec
            velocity = [np.sqrt(agent.vx_vec[i] ** 2 + agent.vy_vec[i] ** 2) for i in range(len(agent.vx_vec))]

            # get feature values
            features['episode'].append(inx)
            features['id'].append(agent_id)
            features['long_acc_max'].append(feature_extractor.get_max_value(long_acc))
            features['long_acc_fc'].append(feature_extractor.get_frequency_centroid(long_acc))
            features['lat_acc_sf'].append(feature_extractor.get_shape_factor(lat_acc))
            features['lat_acc_rms'].append(feature_extractor.get_root_mean_square(lat_acc))
            features['vel_std'].append(feature_extractor.get_standard_devisation(velocity))

    df = pd.DataFrame(features)

    if args.clustering == 'kmeans':
        clustered_dataframe = Clustering.kmeans(df)
        visual = Visualization()
        # first episode, please change accordingly
        visual.plot_clustered_trj_on_map(data_loader.scenario.episodes[0], clustered_dataframe)
    elif args.clustering == 'hierarchical':
        clustered_dataframe = Clustering.hierarchical(df)
    elif args.clustering == 'gmm':
        clustered_dataframe = Clustering.GMM(df)
    else:
        raise 'No clustering method is specified'


if __name__ == '__main__':
    sys.exit(main())
