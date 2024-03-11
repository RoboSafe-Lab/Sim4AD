import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, logistic
import pandas as pd

from sim4ad.util import parse_args
from sim4ad.data.data_loaders import DatasetDataLoader
from sim4ad.path_utils import get_config_path
from sim4ad.clustering.clustering import Clustering, plot_radar_charts, save_clustered_data


def plot_distribution(label, data):
    bins = 30
    # Calculate histogram values
    hist_values, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Fit data to distributions
    normal_params = norm.fit(data)
    logistic_params = logistic.fit(data)

    # Create PDFs for the fitted distributions
    fitted_normal_pdf = norm.pdf(bin_centers, *normal_params)
    fitted_logistic_pdf = logistic.pdf(bin_centers, *logistic_params)

    # Plot histogram and PDFs
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='blue', label='ITTC')
    plt.plot(bin_centers, fitted_normal_pdf, 'b-', label='N-fitting')
    plt.plot(bin_centers, fitted_logistic_pdf, 'g-', label='Logistic')

    # Customize the plot
    plt.xlabel(label)
    plt.ylabel('Normalized PDF')
    plt.legend()
    plt.grid(True, which='both', axis='both', color='black', linestyle='--', linewidth=0.5)

    # For the CDF plot:
    # Calculate the empirical CDF
    data_sorted = np.sort(data)
    cdf_empirical = np.arange(1, len(data) + 1) / len(data)

    # Calculate the CDFs for the fitted parameters
    cdf_norm = norm.cdf(data_sorted, *normal_params)
    cdf_logistic = logistic.cdf(data_sorted, *logistic_params)

    # Plot the empirical CDF and the fitted CDFs
    plt.figure(figsize=(6, 4))
    plt.plot(cdf_empirical, data_sorted,  'ko', markersize=2, alpha=0.5, label='Empirical CDF')
    plt.plot(cdf_norm, data_sorted, 'r-', label='N-fitting')
    plt.plot(cdf_logistic, data_sorted, 'g-', label='Logistic')
    # Add T-fitting if you have the parameters or the fitting function

    # Annotate specific percentile values
    percentiles = [0.25, 0.45, 0.65, 0.85, 0.95]
    for percentile in percentiles:
        value = np.percentile(data_sorted, percentile * 100)
        plt.annotate(f'({percentile},{value:.2f})',
                     xy=(percentile, value),
                     xytext=(percentile, value + 0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    plt.xlabel('Percentile Values')
    plt.ylabel(label)
    plt.legend()
    plt.grid(True, which='both', axis='both', color='black', linestyle='--', linewidth=0.5)


def driving_style_matching(cluster_centers, feature_names):
    driver_style = []
    # determine which label belongs to which driving style
    for inx, cluster_center in enumerate(cluster_centers):
        if feature_names[np.argmax(cluster_center)] == 'sp':
            driver_style.append('Cautious')
        elif feature_names[np.argmax(cluster_center)] == 'pdp':
            driver_style.append('Normal')
        elif feature_names[np.argmax(cluster_center)] == 'hrp':
            driver_style.append('Aggressive')
        else:
            raise "Driving style cannot be identified."

    return driver_style


def post_analysis(driver_style, clustered_dataframe) -> pd.DataFrame:
    """Analysis the clustered data"""
    labeled_data = pd.DataFrame()
    grouped_cluster = clustered_dataframe.groupby('label')
    # compute the proportion of different risk levels
    for label, group in grouped_cluster:
        key = driver_style[label]
        processed_group = group.iloc[:, [0, 1, -1]].copy()
        processed_group.iloc[:, -1] = key
        labeled_data = pd.concat([labeled_data, processed_group], ignore_index=True)
        safe_proportion = group['sp'].mean()
        potential_danger_proportion = group['pdp'].mean()
        high_risk_proportion = group['hrp'].mean()
        print(f'{key} drivers number: {len(group)}. Safe proportion: {safe_proportion:.2f},'
              f' potential danger proportion: {potential_danger_proportion:.2f}, '
              f'high risk proportion: {high_risk_proportion:.2f} ')

    return labeled_data


def main():
    args = parse_args()
    data_loader = DatasetDataLoader(get_config_path(args.map))
    data_loader.load()

    # define thresholds for thw and ttc
    ttc_thres = 2.0
    thw_thres = 2.0

    # sp: safety proportion, lrp: low risk proportion, pdp: potential danger proportion, hrp: high risk proportion
    features = {'episode_id': [], 'agent_id': [], 'sp': [], 'lrp': [], 'pdp': [], 'hrp': [], 'label': None}
    ittc = []
    thw = []
    # Traverse all episodes if they belong to the same map
    for episode in data_loader.scenario.episodes:
        for agent_id, agent in episode.agents.items():
            ittc_one_agent, thw_one_agent = ([] for _ in range(2))
            trajectory_proportion = np.zeros(4)
            num = len(agent.ttc_dict_vec)
            for inx in range(num):
                ttc_temp = agent.ttc_dict_vec[inx]['front_ego']
                if ttc_temp is not None and ttc_temp >= 0:
                    ittc_one_agent.append(1 / ttc_temp)

                thw_temp = agent.tth_dict_vec[inx]['front_ego']
                if thw_temp is not None:
                    thw_one_agent.append(thw_temp)

                # trajectory segmentation
                if (ttc_temp is None or ttc_temp <= 0 or 1/ttc_temp <= 1/ttc_thres) and (thw_temp is None or thw_temp >= thw_thres):
                    trajectory_proportion[0] += 1
                elif 1/ttc_temp >= 1/ttc_thres and thw_temp >= thw_thres:
                    trajectory_proportion[1] += 1
                elif 1/ttc_temp <= 1/ttc_thres and thw_temp <= thw_thres:
                    trajectory_proportion[2] += 1
                elif 1/ttc_temp >= 1/ttc_thres and thw_temp <= thw_thres:
                    trajectory_proportion[3] += 1
                else:
                    raise f'Agent {agent_id} at index {inx} cannot be assigned.'

            if ittc_one_agent:
                ittc += ittc_one_agent
            if thw_one_agent:
                thw += thw_one_agent

            proportion = [p/num for p in trajectory_proportion]
            features['episode_id'].append(episode.config.recording_id)
            features['agent_id'].append(agent_id)
            features['sp'].append(proportion[0])
            features['lrp'].append(proportion[1])
            features['pdp'].append(proportion[2])
            features['hrp'].append(proportion[3])

    df = pd.DataFrame(features)
    cluster = Clustering(n_cluster=3)
    clustered_dataframe, cluster_centers = cluster.kmeans(df)
    # calculate important metrics for evaluation
    silhouette, dbi, chi = cluster.evaluation(clustered_dataframe)
    print(f'silhouette value is {silhouette:.2f}')
    print(f'Davies-Bouldin Index is {dbi:.2f}')
    print(f'Calinski-Harabasz Index is {chi:.2f}')

    feature_names = list(features.keys())
    feature_names = feature_names[2:-1]
    driver_style = driving_style_matching(cluster_centers, feature_names)
    labeled_data = post_analysis(driver_style, clustered_dataframe)

    # visualize the radar charts for each cluster
    for inx, cluster_center in enumerate(cluster_centers):
        plot_radar_charts(feature_names, inx, cluster_center, driver_style)

    # plot distribution and cumulative distribution
    plot_distribution(r'iTTC [s$^{-1}$]', ittc)
    plot_distribution('THW [s]', thw)

    plt.show()

    # save labeled data to csv
    save_clustered_data(labeled_data)


if __name__ == '__main__':
    sys.exit(main())
