import pickle
import os
import numpy as np

# 数据类别和文件路径设置
data_styles = ['normal', 'cautious']
data = ['train']
# 初始化奖励存储
normal_storage = []  # 存储 Normal 类型及被重新分类的 Cautious 类型 agent 数据

# Normal 类型的 reward 最小值
normal_min_reward = -2.2381

# 计数满足条件的 cautious 数据
cautious_to_normal_count = 0

# 遍历数据
for data_style in data_styles:
    for d in data:
        file_path = f'scenarios/data/' + d + '/' + 'Normal' + 'appershofen' + data_style + '_agents_mdp.pkl'

        # 跳过空文件
        if os.path.getsize(file_path) == 0:
            continue

        # 加载文件数据
        with open(file_path, 'rb') as file:
            dataset = pickle.load(file)

        # 遍历每个 agent 
        for key, mdp_value in dataset.items():
            if mdp_value is not None:  
                avg_reward = np.mean(mdp_value.rewards)  
                
                
                if data_style == 'cautious' or (data_style == 'cautious' and avg_reward < normal_min_reward):
                
                    normal_storage.append({
                        'agent_id': key,
                        'rewards': mdp_value.rewards,
                        'observations': mdp_value.observations,
                        'actions': mdp_value.actions,
                        'terminals': mdp_value.terminals,
                    })
                    # 如果是符合条件的 Cautious 数据，计数加一
                    if data_style == 'cautious' and avg_reward > 0 :
                        cautious_to_normal_count += 1
            else:
                print(f"Warning: agent id {key} has None value, skipping.")

# 输出存储结果
print(f"Total agents in Normal storage (including cautious-to-normal): {len(normal_storage)}")
print(f"Total cautious agents moved to Normal storage: {cautious_to_normal_count}")
"""
# 保存结果到文件
output_path = 'scenarios/data/train/Normal_Offline_RL.pkl'
with open(output_path, 'wb') as out_file:
    pickle.dump(normal_storage, out_file)
    print(f"Normal storage saved to {output_path}")
"""