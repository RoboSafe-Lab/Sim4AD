import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, logistic
import pandas as pd
from typing import Dict

from sim4ad.util import parse_args
from sim4ad.data.data_loaders import DatasetDataLoader
from sim4ad.path_utils import get_config_path
from sim4ad.clustering.clustering import Clustering, plot_radar_charts, save_clustered_data

def plot_pdf_distribution(label, data):
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
    plt.hist(data, bins=bins, density=True, alpha=0.6, color='blue', label=label)
    plt.plot(bin_centers, fitted_normal_pdf, 'b-', label='N-fitting')
    plt.plot(bin_centers, fitted_logistic_pdf, 'g-', label='Logistic')

    # Customize the plot
    plt.xlabel(label)
    plt.ylabel('Normalized PDF')
    plt.legend()
    plt.grid(True, which='both', axis='both', color='black', linestyle='--', linewidth=0.5)


def plot_cdf_distribution(label, data):
    normal_params = norm.fit(data)
    logistic_params = logistic.fit(data)
    # For the CDF plot:
    # Calculate the empirical CDF
    data_sorted = np.sort(data)
    cdf_empirical = np.arange(1, len(data) + 1) / len(data)

    # Calculate the CDFs for the fitted parameters
    cdf_norm = norm.cdf(data_sorted, *normal_params)
    cdf_logistic = logistic.cdf(data_sorted, *logistic_params)

    # Plot the empirical CDF and the fitted CDFs
    fig = plt.figure(figsize=(6, 4))
    plt.plot(cdf_empirical, data_sorted,  'ko', markersize=2, alpha=0.5, label='Empirical CDF')
    plt.plot(cdf_norm, data_sorted, '#299d8f', label='N-fitting')
    plt.plot(cdf_logistic, data_sorted, '#e66d50', label='Logistic')
    # Add T-fitting if you have the parameters or the fitting function

    # Annotate specific percentile values
    percentiles = [0.25, 0.45, 0.65, 0.85, 0.95]
    for i, percentile in enumerate(percentiles):
        value = np.percentile(data_sorted, percentile * 100)
        
        # Calculate dynamic offset based on data range
        data_range = max(data_sorted) - min(data_sorted)
        y_offset = data_range * 0.08  # 8% of data range for vertical offset
        
        # Alternate annotation positions to avoid overlap
        if i % 2 == 0:
            xytext_pos = (percentile, value + y_offset)
            va_align = 'bottom'
        else:
            xytext_pos = (percentile, value - y_offset)
            va_align = 'top'
        
        plt.annotate(f'P{int(percentile*100)}: {value:.2f}',
                     xy=(percentile, value),
                     xytext=xytext_pos,
                     ha='center', va=va_align,
                     fontsize=12, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                     arrowprops=dict(arrowstyle='->', color='black', lw=1))

    plt.xlabel('Percentile Values')
    plt.ylabel(label)
    plt.legend()
    plt.grid(True, which='both', axis='both', color='black', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    return fig

def driving_style_matching(cluster_centers, feature_names):
    driver_style = []
    # determine which label belongs to which driving style
    for inx, cluster_center in enumerate(cluster_centers):
        if feature_names[np.argmax(cluster_center)] == 'LRP':
            driver_style.append('Cautious')
        elif feature_names[np.argmax(cluster_center)] == 'MRP':
            driver_style.append('Normal')
        elif feature_names[np.argmax(cluster_center)] == 'HRP':
            driver_style.append('Aggressive')
        else:
            raise "Driving style cannot be identified."

    return driver_style


def post_analysis(driver_style, clustered_dataframe) -> Dict:
    """Analysis the clustered data"""
    labeled_data = {}
    grouped_cluster = clustered_dataframe.groupby('label')
    
    # Create a dictionary to store results by driving style
    style_results = {}
    
    # compute the proportion of different risk levels
    for label, group in grouped_cluster:
        ds = driver_style[label]
        for index, row in group.iterrows():
            key = row[0] + '/' + row[1]
            labeled_data[key] = ds
        safe_proportion = group['LRP'].mean()
        potential_danger_proportion = group['MRP'].mean()
        high_risk_proportion = group['HRP'].mean()
        
        # Store results by driving style for consistent ordering
        style_results[ds] = {
            'count': len(group),
            'LRP': safe_proportion,
            'MRP': potential_danger_proportion,
            'HRP': high_risk_proportion
        }
    
    # Print results in consistent order: Cautious, Normal, Aggressive
    preferred_order = ['Cautious', 'Normal', 'Aggressive']
    for style in preferred_order:
        if style in style_results:
            result = style_results[style]
            print(f'{style} drivers number: {result["count"]}. LRP: {result["LRP"]:.2f},'
                  f' MRP: {result["MRP"]:.2f}, '
                  f'HRP: {result["HRP"]:.2f} ')

    return labeled_data

def plot_ittc_thw_clusters(ittc_thw_data, clustered_dataframe, driver_style):
    """Plot ITTC vs THW scatter plot with different colors for each cluster"""
    plt.figure(figsize=(6, 4))
    
    # Define colors for each cluster
    colors = ['#299d8f', '#e7c66b', '#e66d50', 'purple', 'brown', 'pink', 'gray']
    markers = ['^', '^', '^', 'D', 'v', '<', '>', 'p']
    
    # Create a mapping from episode_id/agent_id to cluster label
    agent_to_cluster = {}
    for index, row in clustered_dataframe.iterrows():
        key = str(row['episode_id']) + '/' + str(row['agent_id'])
        agent_to_cluster[key] = row['label']
    
    # Create separate lists for each cluster
    cluster_data = {}
    for i, (episode_id, agent_id, ittc_val, thw_val) in enumerate(zip(
        ittc_thw_data['episode_id'], ittc_thw_data['agent_id'], 
        ittc_thw_data['ittc'], ittc_thw_data['thw'])):
        
        key = str(episode_id) + '/' + str(agent_id)
        if key in agent_to_cluster:
            cluster_label = agent_to_cluster[key]
            if cluster_label not in cluster_data:
                cluster_data[cluster_label] = {'ittc': [], 'thw': []}
            cluster_data[cluster_label]['ittc'].append(ittc_val)
            cluster_data[cluster_label]['thw'].append(thw_val)
    
    # Plot each cluster with different color
    for i, (label, data) in enumerate(cluster_data.items()):
        plt.scatter(data['ittc'], data['thw'], 
                   c=colors[i % len(colors)], 
                   marker=markers[i % len(markers)],
                   label=f'{driver_style[label]}',
                   alpha=0.6, s=30)
    
    plt.xlabel(r'iTTC [s$^{-1}$]', fontsize=14)
    plt.ylabel('THW [s]', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add threshold lines for reference
    plt.axhline(y=2.0, color='black', linestyle='--', alpha=0.5, label='THW threshold (2.0s)')
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='ITTC threshold (0.5s⁻¹)')
    plt.savefig('kmeans-clusters.png', dpi=300) 
    

def plot_cluster_distribution_comparison(methods_metrics):
    """Compare cluster size distributions across methods using histograms"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    methods = list(methods_metrics.keys())
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        clustered_df = methods_metrics[method]['clustered_df']
        cluster_centers = methods_metrics[method]['cluster_centers']
        
        # Get driving styles
        feature_names = ['LRP', 'MRP', 'HRP']
        driver_style = driving_style_matching(cluster_centers, feature_names)
        
        # Count agents per cluster and organize by driving style
        cluster_counts = clustered_df['label'].value_counts()
        
        # Create ordered data by driving style (Cautious, Normal, Aggressive)
        preferred_order = ['Cautious', 'Normal', 'Aggressive']
        ordered_labels = []
        ordered_counts = []
        ordered_colors = []
        
        for style in preferred_order:
            # Find which cluster ID corresponds to this driving style
            for cluster_id, style_name in enumerate(driver_style):
                if style_name == style and cluster_id in cluster_counts.index:
                    ordered_labels.append(style)
                    ordered_counts.append(cluster_counts[cluster_id])
                    if style == 'Cautious':
                        ordered_colors.append('#299d8f')
                    elif style == 'Normal':
                        ordered_colors.append('#e7c66b')
                    else:  # Aggressive
                        ordered_colors.append('#e66d50')
                    break
        
        # Create histogram/bar chart with ordered data
        bars = ax.bar(ordered_labels, ordered_counts, color=ordered_colors, 
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add count annotations on top of bars
        total_count = sum(ordered_counts)
        for i, (bar, count) in enumerate(zip(bars, ordered_counts)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + max(ordered_counts)*0.01,
                   f'{count}\n({count/total_count*100:.1f}%)', 
                   ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        ax.set_ylabel('Numbers', fontsize=14)
        ax.set_title(f'{method.upper()}', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Set y-axis to start from 0 and add some headroom
        if ordered_counts:
            ax.set_ylim(0, max(ordered_counts) * 1.15)
    
    plt.tight_layout()
    return fig


def plot_cluster_separation_quality(methods_metrics):
    """Visualize cluster separation quality using within-cluster vs between-cluster distances"""
    methods = list(methods_metrics.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        clustered_df = methods_metrics[method]['clustered_df']
        
        # Calculate within-cluster and between-cluster distances
        within_distances = []
        between_distances = []
        
        features = clustered_df[['LRP', 'MRP', 'HRP']].values
        labels = clustered_df['label'].values
        
        # Within-cluster distances
        for label in np.unique(labels):
            cluster_points = features[labels == label]
            if len(cluster_points) > 1:
                center = cluster_points.mean(axis=0)
                distances = np.sqrt(((cluster_points - center) ** 2).sum(axis=1))
                within_distances.extend(distances)
        
        # Between-cluster distances (sample for efficiency)
        n_samples = min(1000, len(features))
        sample_indices = np.random.choice(len(features), n_samples, replace=False)
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if labels[sample_indices[i]] != labels[sample_indices[j]]:
                    dist = np.sqrt(((features[sample_indices[i]] - features[sample_indices[j]]) ** 2).sum())
                    between_distances.append(dist)
        
        # Plot histograms
        ax.hist(within_distances, bins=30, alpha=0.7, label='Within-cluster', 
                color='#e66d50', density=True)  # Red/Orange consistent with aggressive
        ax.hist(between_distances, bins=30, alpha=0.7, label='Between-cluster', 
                color='#299d8f', density=True)  # Teal consistent with cautious
        
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        ax.set_title(f'{method.upper()}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add separation quality metric
        within_mean = np.mean(within_distances) if within_distances else 0
        between_mean = np.mean(between_distances) if between_distances else 0
        separation_ratio = between_mean / within_mean if within_mean > 0 else 0
        
        ax.text(0.05, 0.95, f'Separation Ratio: {separation_ratio:.2f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def plot_metrics_hist(methods_metrics):
    """Plot histogram/bar chart comparing clustering evaluation metrics across methods"""
    methods = list(methods_metrics.keys())
    silhouette_scores = [methods_metrics[method]['silhouette'] for method in methods]
    dbi_scores = [methods_metrics[method]['dbi'] for method in methods]
    chi_scores = [methods_metrics[method]['chi'] for method in methods]
    
    # Create subplots for the three metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#299d8f', '#e7c66b', '#e66d50']
    
    # Silhouette Score (higher is better)
    ax1 = axes[0]
    bars1 = ax1.bar(methods, silhouette_scores, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Silhouette Score\n(Higher is Better)', fontsize=14)
    ax1.set_ylabel('Silhouette Score', fontsize=14)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars1, silhouette_scores)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Davies-Bouldin Index (lower is better)
    ax2 = axes[1]
    bars2 = ax2.bar(methods, dbi_scores, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title('Davies-Bouldin Index\n(Lower is Better)', fontsize=14)
    ax2.set_ylabel('Davies-Bouldin Index', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars2, dbi_scores)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(dbi_scores)*0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Calinski-Harabasz Index (higher is better)
    ax3 = axes[2]
    bars3 = ax3.bar(methods, chi_scores, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_title('Calinski-Harabasz Index\n(Higher is Better)', fontsize=14)
    ax3.set_ylabel('Calinski-Harabasz Index', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars3, chi_scores)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(chi_scores)*0.01, 
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Highlight best performing method for each metric
    best_silhouette_idx = np.argmax(silhouette_scores)
    best_dbi_idx = np.argmin(dbi_scores)  # Lower is better for DBI
    best_chi_idx = np.argmax(chi_scores)
    
    bars1[best_silhouette_idx].set_edgecolor('red')
    bars1[best_silhouette_idx].set_linewidth(3)
    
    bars2[best_dbi_idx].set_edgecolor('red')
    bars2[best_dbi_idx].set_linewidth(3)
    
    bars3[best_chi_idx].set_edgecolor('red')
    bars3[best_chi_idx].set_linewidth(3)
    
    plt.tight_layout()
    
    # Print summary
    print("\n=== CLUSTERING METHODS COMPARISON SUMMARY ===")
    for i, method in enumerate(methods):
        print(f"{method.upper():>12}: Silhouette={silhouette_scores[i]:.3f}, "
              f"DBI={dbi_scores[i]:.3f}, CHI={chi_scores[i]:.1f}")
    
    print(f"\nBest Methods:")
    print(f"  Silhouette Score: {methods[best_silhouette_idx].upper()} ({silhouette_scores[best_silhouette_idx]:.3f})")
    print(f"  Davies-Bouldin:   {methods[best_dbi_idx].upper()} ({dbi_scores[best_dbi_idx]:.3f})")
    print(f"  Calinski-Harabasz: {methods[best_chi_idx].upper()} ({chi_scores[best_chi_idx]:.1f})")
    
    return fig


def main():
    args = parse_args()
    data_loader = DatasetDataLoader(get_config_path(args.map))
    data_loader.load()

    # define thresholds for thw and ttc
    ttc_thres = 2.0
    thw_thres = 2.0

    # LRP: low risk proportion, MRP: medium risk proportion, HRP: high risk proportion
    features = {'episode_id': [], 'agent_id': [], 'LRP': [], 'MRP': [], 'HRP': [], 'label': None}
    ittc_thw = {'episode_id': [], 'agent_id': [], 'ittc': [], 'thw': []}

    ittc_dist = []
    thw_dist = []
    # Traverse all episodes if they belong to the same map
    for episode in data_loader.scenario.episodes:
        for agent_id, agent in episode.agents.items():
            ittc_one_agent, thw_one_agent = ([] for _ in range(2))
            trajectory_proportion = np.zeros(3)
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
                elif 1/ttc_temp <= 1/ttc_thres and thw_temp <= thw_thres:
                    trajectory_proportion[1] += 1
                elif 1/ttc_temp >= 1/ttc_thres and thw_temp <= thw_thres:
                    trajectory_proportion[2] += 1
                else:
                    raise f'Agent {agent_id} at index {inx} cannot be assigned.'

            if ittc_one_agent:
                ittc_dist += ittc_one_agent
            if thw_one_agent:
                thw_dist += thw_one_agent
            
            # Only include agents that have BOTH ITTC and THW data for the plot
            if ittc_one_agent and thw_one_agent:
                ittc_thw['ittc'].append(np.max(ittc_one_agent))
                ittc_thw['thw'].append(np.min(thw_one_agent))
                ittc_thw['episode_id'].append(episode.config.recording_id)
                ittc_thw['agent_id'].append(agent_id)
                
            proportion = [p/num for p in trajectory_proportion]
            features['episode_id'].append(episode.config.recording_id)
            features['agent_id'].append(agent_id)
            features['LRP'].append(proportion[0])
            features['MRP'].append(proportion[1])
            features['HRP'].append(proportion[2])


    df = pd.DataFrame(features)
    feature_names = list(features.keys())
    feature_names = feature_names[2:-1]
    
    # Methods comparison - collect all metrics
    methods = ['kmeans', 'hierarchical', 'GMM']
    methods_metrics = {}
    
    for method in methods:
        cluster = Clustering(n_cluster=3)
        if method == 'kmeans':
            clustered_df, cluster_centers = cluster.kmeans(df.copy())
        elif method == 'hierarchical':
            clustered_df, cluster_centers = cluster.hierarchical(df.copy())
        elif method == 'GMM':
            clustered_df, cluster_centers = cluster.GMM(df.copy())

        # calculate important metrics for evaluation
        silhouette, dbi, chi = cluster.evaluation(clustered_df)
        
        # Store metrics for comparison
        methods_metrics[method] = {
            'silhouette': silhouette,
            'dbi': dbi,
            'chi': chi,
            'clustered_df': clustered_df,
            'cluster_centers': cluster_centers
        }
        
        driver_style = driving_style_matching(cluster_centers, feature_names)
        
        print(f'\n{method.upper()} Results:')
        print(f'Silhouette Score: {silhouette:.3f}')
        print(f'Davies-Bouldin Index: {dbi:.3f}')
        print(f'Calinski-Harabasz Index: {chi:.1f}')
        
        driver_style = driving_style_matching(cluster_centers, feature_names)
        labeled_data = post_analysis(driver_style, clustered_df)

    # 1. Plot histogram for metrics comparison across all methods
    print("=== CREATING CLUSTERING METRICS COMPARISON ===")
    fig_metrics = plot_metrics_hist(methods_metrics)
    plt.savefig('clustering_metrics_comparison.png', dpi=300, bbox_inches='tight')
    
    # 2. Cluster distribution comparison (pie charts)
    print("=== CREATING CLUSTER DISTRIBUTION COMPARISON ===")
    fig_distribution = plot_cluster_distribution_comparison(methods_metrics)
    plt.savefig('cluster_distribution_comparison.png', dpi=300, bbox_inches='tight')

    # 3. Cluster separation quality
    print("=== CREATING CLUSTER SEPARATION QUALITY COMPARISON ===")
    fig_separation = plot_cluster_separation_quality(methods_metrics)
    plt.savefig('cluster_separation_quality.png', dpi=300, bbox_inches='tight')

    # 4. Plot cumulative distribution
    print("=== CREATING CDF DISTRIBUTION PLOTS ===")
    fig_ittc = plot_cdf_distribution(r'iTTC [s$^{-1}$]', ittc_dist)
    plt.savefig('cdf_distribution_ittc.png', dpi=300, bbox_inches='tight')
    fig_thw = plot_cdf_distribution('THW [s]', thw_dist)
    plt.savefig('cdf_distribution_thw.png', dpi=300, bbox_inches='tight')

    # Plot iTTC vs THW with cluster colors for k-means method only
    if 'kmeans' in methods_metrics:
        print("=== CREATING iTTC vs THW SCATTER PLOT FOR K-MEANS ===")
        kmeans_data = methods_metrics['kmeans']
        clustered_df = kmeans_data['clustered_df']
        cluster_centers = kmeans_data['cluster_centers']
        driver_style = driving_style_matching(cluster_centers, feature_names)
        plot_ittc_thw_clusters(ittc_thw, clustered_df, driver_style)
        print("K-means scatter plot saved as 'kmeans-clusters.png'")

    plt.show()

    # save labeled data to json file for the best method
    # save_clustered_data(args.map, labeled_data)


if __name__ == '__main__':
    sys.exit(main())
