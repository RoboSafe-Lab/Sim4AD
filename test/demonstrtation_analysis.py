import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sim4ad.offlinerlenv.td3bc_automatum import qlearning_dataset


def plot_histograms(data):
    for key, value in data.items():
        if key == 'actions':
            plt.figure(figsize=(10, 4))
            if value.ndim > 1:
                for i in range(value.shape[1]):
                    plt.subplot(1, value.shape[1], i + 1)
                    sns.histplot(value[:, i], bins=30, kde=True)
                    plt.title(f'{key} Dimension {i+1}')
                    print(f'max value of Dimension {i+1} is: {max(value[:, i])}')
                    print(f'min value of Dimension {i + 1} is: {min(value[:, i])}')
                    print(f'mean value of Dimension {i + 1} is: {np.mean(value[:, i])}')
            else:
                sns.histplot(value, bins=30, kde=True)
                plt.title(key)
            plt.tight_layout()
            plt.show()
        elif key == 'rewards':
            # reward scaling and normalization
            print(f'max value of reward is: {max(value)}')
            print(f'min value of reward is: {min(value)}')
            print(f'mean value of reward is: {np.mean(value)}')
            print(f'std value of reward is: {np.std(value)}')
            value = (value - np.mean(value)) / np.std(value)
            #value = np.tanh(value)
            #print(f'max value of reward is: {max(value)}')
            #print(f'min value of reward is: {min(value)}')
            #countd
            threshold_1 = -10.1   # 设置异常值判断的阈值
            threshold_2 = 10.1
            anomalous_indices = [i for i, r in enumerate(value) if r > threshold_2 ]
            print(f"Number of anomalous rewards: {len(anomalous_indices)}")
            # 计算并打印异常值的比例
            total_rewards = len(value)  # 总奖励数
            anomalous_ratio = len(anomalous_indices) / total_rewards * 100  # 异常值占比（百分比）
            print(f"Percentage of anomalous rewards: {anomalous_ratio:.3f}%")

            plt.figure(figsize=(10, 4))
            sns.histplot(value, bins=30, kde=True)
            plt.title(key)
            plt.tight_layout()
            plt.show()


driving_style = 'Normal'
data = ['train', 'test']
# load demonstration data
all_demonstrations = {'All': [], 'clustered': []}
for d in data:
    with open('scenarios/data/' + d + '/' + driving_style + 'appershofen_demonstration.pkl', 'rb') as file:
        dataset = pickle.load(file)
        all_demonstrations['All'].extend(dataset.get('All', []))
        all_demonstrations['clustered'].extend(dataset.get('clustered', []))

agents_data = []
for agent_mdp in all_demonstrations['clustered']:
    agent_data = qlearning_dataset(dataset=agent_mdp)
    agents_data.append(agent_data)
keys = agents_data[0].keys()
data = {key: np.concatenate([agent_data[key] for agent_data in agents_data]) for key in keys}

plot_histograms(data)


