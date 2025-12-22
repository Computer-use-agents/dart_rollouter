import json
import pandas as pd
import os

# ================= 配置区域 =================
INPUT_FILE = 'test_nogdrive.json'      # 输入的JSON文件名
OUTPUT_FILE = 'test_nogdrive.xlsx'   # 输出的Excel文件名

with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 将数据转换为扁平化的字典列表
rows = []
for category, uuid_list in data.items():
    for uuid in uuid_list:
        rows.append({"Category": category, "UUID": uuid})

# 转换为 DataFrame
df = pd.DataFrame(rows)

# 导出
df.to_excel(OUTPUT_FILE, index=False)
print(f"转换完成，这种格式最适合做数据透视表！")