"""
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
"""
# 查看 dataset 的类型和内容
print("Dataset type:", type(dataset))
print("Dataset content:", dataset)
"""
import pickle
import os

# 数据类别和文件路径设置
data_styles = ['cautious','normal','aggressive']
data = ['train', 'test']
# 初始化奖励列表
rewards_normal = []
rewards_aggressive = []
rewards_cautious = []

target_reward = 1.8
tolerance = 0.02 
matching_keys = []
# 遍历数据
for data_style in data_styles:
    for d in data:
        file_path = f'scenarios/data/' + d + '/' + 'Normal' + 'appershofen' + data_style + '_agents_mdp.pkl'
        #file_path = f'scenarios/data/' + d + '/' + 'Normal' + 'appershofen_demonstration.pkl'

        if os.path.getsize(file_path) == 0:
            continue
        # 加载数据
        with open(file_path, 'rb') as file:
            dataset = pickle.load(file)
        
        # 根据数据类型选择对应的奖励列表
        if data_style == 'normal':
            rewards_list = rewards_normal
        elif data_style == 'aggressive':
            rewards_list = rewards_aggressive
        elif data_style == 'cautious':
            rewards_list = rewards_cautious

        # 提取奖励数据
        for key, mdp_value in dataset.items():
            if mdp_value is not None:  # 确保mdp_value不为None
                rewards_list.extend(mdp_value.rewards)
                # 遍历mdp_value.rewards中的每个奖励值，检查与target_reward的差值
                for reward in mdp_value.rewards:
                    #if abs(reward - target_reward) < tolerance:
                    if reward > 2.0 :
                        matching_keys.append(key)  # 将符合条件的agent id添加到matching_keys列表中
                        break  # 如果找到一个匹配的奖励，可以跳出当前奖励循环
            else:
                print(f"Warning: agent id {key} has None value, skipping.")    
print(f"Keys corresponding to reward {target_reward}: {matching_keys}")

# 计算并输出每组数据的最小值和最大值
#print(f"Normal rewards - Min: {min(rewards_normal)}, Max: {max(rewards_normal)}")
#print(f"Aggressive rewards - Min: {min(rewards_aggressive)}, Max: {max(rewards_aggressive)}")
#print(f"Cautious rewards - Min: {min(rewards_cautious)}, Max: {max(rewards_cautious)}")





