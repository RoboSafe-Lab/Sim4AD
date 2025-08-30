import matplotlib.pyplot as plt
import numpy as np
import re

# 读取文件内容
with open('q_value.txt', 'r', encoding='utf-8') as file:
    content = file.read()

# 定义一个函数来提取数据
def extract_values(label, text):
    """
    从文本中提取指定标签的数据。
    
    参数:
    label (str): 标签名称，例如 'current_q1_values'
    text (str): 文件内容
    
    返回:
    list of float: 提取到的数值列表
    """
    # 构建正则表达式模式
    pattern = label + r'\s*:\s*(\[\[.*?\]\])'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        data_str = match.group(1)
        # 使用正则表达式提取所有数字，包括科学计数法
        numbers = re.findall(r'[-+]?\d*\.\d+e[+-]?\d+|[-+]?\d+\.\d+|[-+]?\d+', data_str)
        return [float(num) for num in numbers]
    else:
        print(f"未找到标签: {label}")
        return []

# 提取三个部分的数据
current_q1_values = extract_values('current_q1 values', content)
current_q2_values = extract_values('current_q2 values', content)
target_q_values = extract_values('target_q values', content)

# 检查数据是否成功提取
print(f"current_q1_values 数量: {len(current_q1_values)}")
print(f"current_q2_values 数量: {len(current_q2_values)}")
print(f"target_q_values 数量: {len(target_q_values)}")

# 绘制直方图
plt.figure(figsize=(12, 8))

# 绘制 current_q1_values
plt.hist(current_q1_values, bins=100, alpha=0.5, label='current_q1_values', density=True)

# 绘制 current_q2_values
plt.hist(current_q2_values, bins=100, alpha=0.5, label='current_q2_values', density=True)

# 绘制 target_q_values
plt.hist(target_q_values, bins=100, alpha=0.5, label='target_q_values', density=True)

# 添加图例和标签
plt.xlabel('Q value')
plt.ylabel('midu')
plt.title('distribution')
plt.legend()

# 显示图形
plt.show()
