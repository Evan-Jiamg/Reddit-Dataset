import kagglehub
import pandas as pd
import numpy as np
import os

# =====================
# STEP 1: 下載 & 看資料夾結構
# =====================
path = kagglehub.dataset_download(
    "i221113hadiyatanveer/the-pushshift-reddit-dataset-submissions"
)

print("=== 資料夾結構 ===")
for root, dirs, files in os.walk(path):
    level = root.replace(path, '').count(os.sep)
    indent = '  ' * level
    print(f"{indent}{os.path.basename(root)}/")
    for file in files:
        filepath = os.path.join(root, file)
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"{indent}  {file}  ({size_mb:.1f} MB)")

# =====================
# STEP 2: 逐一讀取每個檔案的基本資訊
# =====================
def inspect_file(filepath):
    filename = os.path.basename(filepath)
    ext = os.path.splitext(filepath)[1].lower()
    
    print(f"\n{'='*50}")
    print(f"檔案名稱: {filename}")
    print(f"檔案格式: {ext}")
    
    try:
        if ext == '.csv':
            # 只讀前 5 行
            df = pd.read_csv(filepath, nrows=5)
        elif ext == '.json':
            df = pd.read_json(filepath, lines=True, nrows=5)
        elif ext in ['.parquet', '.pq']:
            df = pd.read_parquet(filepath).head(5)
        else:
            print(f"  ⚠️ 未知格式，略過")
            return
        
        print(f"欄位數量: {len(df.columns)}")
        print(f"欄位名稱: {list(df.columns)}")
        print(f"\n--- 前 5 筆資料 ---")
        print(df)
        print(f"\n--- 資料型別 ---")
        print(df.dtypes)
        
    except Exception as e:
        print(f"  ❌ 讀取失敗: {e}")

# 對每個檔案執行
for root, dirs, files in os.walk(path):
    for file in files:
        filepath = os.path.join(root, file)
        inspect_file(filepath)