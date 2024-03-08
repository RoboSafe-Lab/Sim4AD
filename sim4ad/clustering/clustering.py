from sim4ad.path_utils import get_config_path
from sim4ad.util import parse_args
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns

from sim4ad.data.data_loaders import DatasetDataLoader
from visualization import Visualization


class FeatureExtraction:
    """
    features determined by the paper "Feature selection for driving style and skill clustering using naturalistic
    driving data and driving behavior questionnaire"
    """
    def __init__(self, episodes):
        self._features = {'episode_id': [], 'agent_id': [], 'long_acc_max': [], 'long_acc_fc': [], 'lat_acc_sf': [],
                          'lat_acc_rms': [], 'vel_std': [], 'label': None}
        self.episodes = episodes

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
    def get_standard_deviation(data):
        return np.std(data)

    @staticmethod
    def get_frequency_centroid(data):
        N = len(data)
        # sample frequency obtained from the dataset
        sampling_rate = 30
        dft = np.fft.fft(data)
        magnitude = np.abs(dft)
        freq = np.linspace(0, sampling_rate, N)
        numerator = np.sum(magnitude * freq)
        denominator = np.sum(magnitude)
        frequency_centroid = numerator / denominator if denominator != 0 else 0

        return frequency_centroid

    @staticmethod
    def get_max_value(data):
        return np.max(data)

    def extract_features(self) -> pd.DataFrame:
        """Extract feature values from the dataset for clustering"""
        for episode in self.episodes:
            for agent_id, agent in episode.agents.items():
                long_acc = agent.ax_vec
                lat_acc = agent.ay_vec
                velocity = [np.sqrt(agent.vx_vec[i] ** 2 + agent.vy_vec[i] ** 2) for i in range(len(agent.vx_vec))]

                # get feature values
                self._features['episode_id'].append(episode.config.recording_id)
                self._features['agent_id'].append(agent_id)
                self._features['long_acc_max'].append(self.get_max_value(long_acc))
                self._features['long_acc_fc'].append(self.get_frequency_centroid(long_acc))
                self._features['lat_acc_sf'].append(self.get_shape_factor(lat_acc))
                self._features['lat_acc_rms'].append(self.get_root_mean_square(lat_acc))
                self._features['vel_std'].append(self.get_standard_deviation(velocity))
        df = pd.DataFrame(self._features)

        return df


class Clustering:
    """
    Various clustering methods can be defined here
    Normal, cautious, and aggressive drivers are defined here
    """

    def __init__(self):
        self._n_cluster = 3

    def kmeans(self, dataframe):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(dataframe.iloc[:, 2:-1])
        # replace 3 with your chosen number of clusters
        kmeans = KMeans(init="random", n_init=10, n_clusters=self._n_cluster, max_iter=500)
        kmeans.fit(scaled_features)
        dataframe['label'] = kmeans.labels_

        return dataframe

    def hierarchical(self, dataframe):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(dataframe.iloc[:, 2:-1])

        # Hierarchical clustering
        hc = AgglomerativeClustering(n_clusters=self._n_cluster, metric='euclidean', linkage='ward')
        dataframe['label'] = hc.fit_predict(scaled_features)

        # Plotting dendrogram
        linked = linkage(scaled_features, method='ward')
        plt.figure(figsize=(10, 7))
        dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')

        return dataframe

    def GMM(self, dataframe):
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(dataframe.iloc[:, 2:-1])

        # Create a Gaussian Mixture Model
        gmm = GaussianMixture(n_components=self._n_cluster, random_state=0)
        # Fit the model
        gmm.fit(scaled_features)
        # Predict the cluster for each data point
        dataframe['label'] = gmm.predict(scaled_features)
        clustered_dataframe = dataframe.groupby('label')

        return dataframe

    @staticmethod
    def evaluation(clustered_dataframe):
        """Evaluate the clustered trajectories using some metrics
            Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
        """
        silhouette = silhouette_score(clustered_dataframe.iloc[:, 2:-1], clustered_dataframe.iloc[:, -1])
        dbi = davies_bouldin_score(clustered_dataframe.iloc[:, 2:-1], clustered_dataframe.iloc[:, -1])
        chi = calinski_harabasz_score(clustered_dataframe.iloc[:, 2:-1], clustered_dataframe.iloc[:, -1])

        return silhouette, dbi, chi


def plot_features(y_label, clustered_dataframe):
    """Plot the feature value for clustered trajectories

       Args:
            y_label: the name of the feature
            clustered_dataframe: The entire data after clustering
    """
    plt.figure(figsize=(8, 6))
    # Create a boxplot
    sns.boxplot(x='label', y=y_label, data=clustered_dataframe)
    # Display the plot
    plt.title('Boxplot Grouped by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(y_label)


def plot_clustered_trj_on_map(data_loader, clustered_dataframe):
    """Plot the clustered trajectory on the map"""
    visual = Visualization()
    # first episode, please change accordingly
    visual.plot_clustered_trj_on_map(data_loader.scenario.episodes[0], clustered_dataframe)


def main():
    args = parse_args()
    data_loader = DatasetDataLoader(get_config_path(args.map))
    data_loader.load()

    feature_extractor = FeatureExtraction(data_loader.scenario.episodes)
    # extract feature values from the dataset
    df = feature_extractor.extract_features()

    # begin the clustering
    cluster = Clustering()
    if args.clustering == 'kmeans':
        clustered_dataframe = cluster.kmeans(df)
    elif args.clustering == 'hierarchical':
        clustered_dataframe = cluster.hierarchical(df)
    elif args.clustering == 'gmm':
        clustered_dataframe = cluster.GMM(df)
    else:
        raise 'No clustering method is specified'

    # visualize each feature values after clustering
    plot_features('long_acc_fc', clustered_dataframe)
    plot_features('long_acc_max', clustered_dataframe)
    plot_features('lat_acc_sf', clustered_dataframe)
    plot_features('lat_acc_rms', clustered_dataframe)
    plot_features('vel_std', clustered_dataframe)

    plt.show()

    # calculate important metrics
    silhouette, dbi, chi = cluster.evaluation(clustered_dataframe)
    print('silhouette value is ', silhouette)
    print('Davies-Bouldin Index is ', dbi)
    print('Calinski-Harabasz Index is ', chi)


if __name__ == '__main__':
    sys.exit(main())
