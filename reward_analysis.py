
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sim4ad.offlinerlenv.td3bc_automatum import qlearning_dataset

REMOVED_AGENTS = ['29c74d22-9aa7-442d-b3ca-8a710ef26185', '88849c8f-5765-4898-8833-88589b72b0bd',
                  'c6025d47-2d30-419b-8b18-48ec83a3619c', '0a37851d-eb39-4409-a1ad-c1e6ec313f91','0754a583-8ba1-432f-8272-d6a1b911e689','61c25a4c-e3c6-4dee-83a4-fbd80376ce52',
                  '598a6e25-bc9e-4b0f-bdb1-692f73d3a191','648dd669-20d1-4512-ab1c-bfb4b6b2bc71','bec34161-bddd-4778-9bab-cd4de2b7b8d0',
                  '329b7035-9e9b-4fbf-9cae-4ae562bdd8de','99f93eda-2660-4547-a768-4cdf1cf6913e', 'e6f8081a-67d3-439d-b8fb-1d3d923e19bc',
                  'd5745d85-fddf-4b81-9aaa-b098a2749d48','47139212-b9f4-4acd-b44c-f8c739fb5e2b','394f33f9-3db6-456a-8606-d53a83672158','76e5e8c7-b73a-4b1f-b97e-e3f58534bb5f',
                  '1bd406f4-2b9d-4a5e-98c4-aa1b95afb509','45b55e95-b961-4d57-9260-9b23dd0f0899']

data_styles = ['normal', 'aggressive', 'cautious']
data = ['train']
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
            if key in REMOVED_AGENTS:
                print("stop")
            if mdp_value is not None:  # 检查mdp_value是否为None
                rewards_list.extend(mdp_value.rewards)  
            else:
                print(f"Warning: agent id {key} has None value, skipping.")      
print(f"Normal rewards - Min: {min(rewards_normal)}, Max: {max(rewards_normal)}")
print(f"Aggressive rewards - Min: {min(rewards_aggressive)}, Max: {max(rewards_aggressive)}")
print(f"Cautious rewards - Min: {min(rewards_cautious)}, Max: {max(rewards_cautious)}")

plt.figure(figsize=(10, 6))
#plt.hist(rewards_normal, bins=50, edgecolor='black', alpha=0.7, density=True)
plt.hist(rewards_normal, bins=60, edgecolor='black', alpha=0.7, density=True, label='Normal', color='blue')
plt.hist(rewards_aggressive, bins=60, edgecolor='black', alpha=0.7, density=True, label='Aggressive', color='red')
plt.hist(rewards_cautious, bins=60, edgecolor='black', alpha=0.7, density=True, label='Cautious', color='green')

plt.xlabel('Reward')
plt.ylabel('Probability Density')

plt.title('Distribution of Rewards (Probability Density)')

#plt.xlim(-4., 2.0)  
#plt.xticks(np.arange(-4.0, 0.5, 2.0))  
plt.legend()
plt.show()



"""
import pickle
import os

# 数据类别和文件路径设置
data_styles = ['normal','cautious','aggressive']
data = ['train']
# 初始化奖励列表
rewards_normal = []
rewards_aggressive = []
rewards_cautious = []

target_reward = 6.8
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
                    if reward <-2.238 :
                        matching_keys.append(key)  # 将符合条件的agent id添加到matching_keys列表中
                        break  # 如果找到一个匹配的奖励，可以跳出当前奖励循环
            else:
                print(f"Warning: agent id {key} has None value, skipping.")    
print(f"Keys corresponding to reward {target_reward}: {matching_keys}")

# 计算并输出每组数据的最小值和最大值
print(f"Normal rewards - Min: {min(rewards_normal)}, Max: {max(rewards_normal)}")
print(f"Aggressive rewards - Min: {min(rewards_aggressive)}, Max: {max(rewards_aggressive)}")
print(f"Cautious rewards - Min: {min(rewards_cautious)}, Max: {max(rewards_cautious)}")
"""
"""
# 在打印最小、最大值的同时，也打印平均值（如果列表不为空）
def print_stats(name, rewards_list):
    if len(rewards_list) > 0:
        print(f"{name} rewards - Min: {min(rewards_list)}, Max: {max(rewards_list)}, "
              f"Mean: {np.mean(rewards_list):.4f}")
    else:
        print(f"{name} rewards - No data available.")

print_stats("Normal", rewards_normal)
print_stats("Aggressive", rewards_aggressive)
print_stats("Cautious", rewards_cautious)
"""


