import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, logistic

from sim4ad.util import parse_args
from sim4ad.data.data_loaders import DatasetDataLoader
from sim4ad.path_utils import get_config_path


def plot_distribution(data):
    # Calculate histogram values
    hist_values, bin_edges = np.histogram(data, bins=30, density=True)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    # Fit data to distributions
    normal_params = norm.fit(data)
    logistic_params = logistic.fit(data)

    # Create PDFs for the fitted distributions
    fitted_normal_pdf = norm.pdf(bin_centers, *normal_params)
    fitted_logistic_pdf = logistic.pdf(bin_centers, *logistic_params)

    # Plot histogram and PDFs
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.6, color='blue', label='ITTC')
    plt.plot(bin_centers, fitted_normal_pdf, 'r-', label='N-fitting')
    plt.plot(bin_centers, fitted_logistic_pdf, 'g-', label='Logistic')

    # Customize the plot
    plt.xlabel('ITTC [s^-1]')
    plt.ylabel('Normalized PDF')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # For the CDF plot:
    # Calculate the empirical CDF
    data_sorted = np.sort(data)
    cdf_empirical = np.arange(1, len(data) + 1) / len(data)

    # Calculate the CDFs for the fitted parameters
    cdf_norm = norm.cdf(data_sorted, *normal_params)
    cdf_logistic = logistic.cdf(data_sorted, *logistic_params)

    # Plot the empirical CDF and the fitted CDFs
    plt.figure(figsize=(8, 5))
    plt.plot(data_sorted, cdf_empirical, 'ko', markersize=2, alpha=0.5, label='Empirical CDF')
    plt.plot(data_sorted, cdf_norm, 'r-', label='N-fitting')
    plt.plot(data_sorted, cdf_logistic, 'g-', label='Logistic')
    # Add T-fitting if you have the parameters or the fitting function

    # Annotate specific percentile values
    percentiles = [0.25, 0.45, 0.65, 0.85, 0.95]
    for percentile in percentiles:
        value = np.percentile(data, percentile * 100)
        plt.annotate(f'({percentile},{value:.2f})',
                     xy=(percentile, value),
                     xytext=(percentile, value + 0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    plt.xlabel('Percentile Values')
    plt.ylabel('ITTC [s^-1]')
    plt.legend()
    plt.show()


def main():
    args = parse_args()
    data_loader = DatasetDataLoader(get_config_path(args.map))
    data_loader.load()

    features = {'episode_id': [], 'agent_id': [], 'ittc_mean': [], 'thw_mean': []}
    ittc_mean = []
    thw_mean = []
    # Traverse all episodes if they belong to the same map
    for episode in data_loader.scenario.episodes:
        for agent_id, agent in episode.agents.items():
            ittc_one_agent = []
            thw_one_agent = []
            for inx in range(len(agent.ttc_dict_vec)):
                ttc_temp = agent.ttc_dict_vec[inx]['front_ego']
                if ttc_temp is not None and ttc_temp >= 0:
                    ittc_one_agent.append(1 / ttc_temp)
                thw_temp = agent.tth_dict_vec[inx]['front_ego']
                if thw_temp is not None and thw_temp >= 0:
                    thw_one_agent.append(thw_temp)

            if ittc_one_agent:
                ittc_mean += ittc_one_agent
            if thw_one_agent:
                thw_mean += thw_one_agent

    # plot distribution and cumulative distribution
    plot_distribution(ittc_mean)


if __name__ == '__main__':
    sys.exit(main())
