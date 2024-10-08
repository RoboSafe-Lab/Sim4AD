"""
import pickle

#.pkl 文件路径
file_path = 'D:\\IRLcode\\Sim4AD\\Aggressive_hw-a9-appershofen-001-d8087340-8287-46b6-9612-869b09e68448_buffer.pkl'

# 打开并加载 .pkl 文件
with open(file_path, 'rb') as file:
    human_traj_features, buffer = pickle.load(file)

# 输出文件
output_file = 'D:\\IRLcode\\Sim4AD\\aggressive_output.txt'

# 将数写入到文件
with open(output_file, 'w') as f:
    f.write("Human Trajectory Features:\n")
    f.write(str(human_traj_features))  
    f.write("\n\nBuffer:\n")
    f.write(str(buffer))  

print(f"Data has been written to {output_file}")
"""
import pickle

# .pkl 文件路径
file_path = 'D:\\IRLcode\\Sim4AD\\Aggressive_hw-a9-appershofen-001-d8087340-8287-46b6-9612-869b09e68448_buffer.pkl'

# 打开并加载 .pkl 文件
with open(file_path, 'rb') as file:
    human_traj_features, buffer = pickle.load(file)

# 输出文件
output_file = 'D:\\IRLcode\\Sim4AD\\aggressive_output_limited.txt'

# 限制输出的条数
max_features = 100  # 限制 human_traj_features 输出的组数
max_buffer = 100    # 限制 buffer 输出的组数

# 将数写入到文件（限制条数）
with open(output_file, 'w') as f:
    f.write("Human Trajectory Features (Limited to 10 sets):\n")
    for i, feature in enumerate(human_traj_features):
        if i >= max_features:
            break
        f.write(str(feature) + '\n')

    f.write("\n\nBuffer (Limited to 10 sets):\n")
    for i, buf in enumerate(buffer):
        if i >= max_buffer:
            break
        f.write(str(buf) + '\n')

print(f"Data has been limited and written to {output_file}")
