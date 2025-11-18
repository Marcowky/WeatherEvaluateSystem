import json
import os
import yaml
import pandas as pd

from .file_timestamp import get_timestamp


def path_preprocess(path: str) -> str:
    # 若已存在该路径，则报错
    if os.path.exists(path):
        old_path = path
        base, ext = os.path.splitext(path)
        path = f"{base}_{get_timestamp()}{ext}"
        print(f"File {old_path} already exists. Adding timestamp to filename {path}.")
        
    # 创建目录
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def load_json(file_path: str) -> dict:
    return json.load(open(file_path, 'r'))

def save_json(data, file_path: str) -> str:
    file_path = path_preprocess(file_path)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return file_path

def load_jsonl(file_path: str) -> list:
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
        
def save_jsonl(data: list, file_path: str) -> str:
    file_path = path_preprocess(file_path)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    return file_path

def load_yaml(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
    
def str_to_json(json_str: str) -> dict:
    return json.loads(json_str)

def get_geo_division():
    geo_division_path = 'data/station_info/地理划分_去除空列.csv'
    geo_division_df = pd.read_csv(geo_division_path, dtype=str)
    return geo_division_df