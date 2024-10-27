import pickle

# route
file_path = 'Aggressive_hw-a9-appershofen-008-44cb097b-ce86-4d2d-b509-0e0c5b5b7ad5_buffer.pkl'
output_file = 'data_output.txt'

# load data
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# .txt
with open(output_file, 'w') as out_file:
    out_file.write(str(data))
    
print(f"Data saved to {output_file}")
