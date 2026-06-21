import json
import os

# 原始文件路径
input_file = 'medical_records.json'
# 输出目录
output_dir = 'data'

# 创建data文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print('正在读取原始数据（JSON Lines格式）...')

# 逐行读取，每行是一个JSON对象
data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f'跳过无效行：{e}')
                continue

total = len(data)
print(f'总数据量：{total} 条')

# 拆分成4份
num_parts = 16
chunk_size = total // num_parts + 1

for i in range(num_parts):
    start = i * chunk_size
    end = min((i + 1) * chunk_size, total)
    chunk = data[start:end]
    
    output_file = os.path.join(output_dir, f'medical_records_part{i+1}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunk, f, ensure_ascii=False, indent=2)
    
    print(f'part{i+1}.json：{len(chunk)} 条，保存到 {output_file}')

print(f'✅ 拆分完成！共生成 {num_parts} 个文件，保存在 data/ 文件夹中')