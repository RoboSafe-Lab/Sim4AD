import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sim4ad.offlinerlenv.td3bc_automatum import qlearning_dataset

data_styles = ['normal', 'aggressive', 'cautious']
data = ['train', 'test']
# load demonstration data
rewards_normal = []
rewards_aggressive = []
rewards_cautious = []
for data_style in data_styles:
    for d in data:
        file_path = f'scenarios/data/' + d + '/' + 'Normal' + 'appershofen' + data_style + '_agents_mdp.pkl'
        if os.path.getsize(file_path) == 0:
            continue
        with open('scenarios/data/' + d + '/' + 'Normal' + 'appershofen' + data_style + '_agents_mdp.pkl', 'rb') as file:
            dataset = pickle.load(file)
        if data_style == 'normal':
            rewards_list = rewards_normal
        elif data_style == 'aggressive':
            rewards_list = rewards_aggressive
        elif data_style == 'cautious':
            rewards_list = rewards_cautious

        for key, mdp_value in dataset.items():
            if mdp_value is not None:  # 检查mdp_value是否为None
                rewards_list.extend(mdp_value.rewards)  
            else:
                print(f"Warning: agent id {key} has None value, skipping.")      


plt.figure(figsize=(10, 6))
#plt.hist(rewards_normal, bins=50, edgecolor='black', alpha=0.7, density=True)
plt.hist(rewards_normal, bins=60, edgecolor='black', alpha=0.7, density=True, label='Normal', color='blue')
plt.hist(rewards_aggressive, bins=60, edgecolor='black', alpha=0.7, density=True, label='Aggressive', color='red')
plt.hist(rewards_cautious, bins=60, edgecolor='black', alpha=0.7, density=True, label='Cautious', color='green')

plt.xlabel('Reward')
plt.ylabel('Probability Density')

plt.title('Distribution of Rewards (Probability Density)')

plt.xlim(-2., 2.0)  
plt.xticks(np.arange(-2.5, 2.0, 0.5))  
plt.legend()
plt.show()
"""
# 查看 dataset 的类型和内容
print("Dataset type:", type(dataset))
print("Dataset content:", dataset)
"""



