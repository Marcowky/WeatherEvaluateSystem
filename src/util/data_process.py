import json
import os
import yaml

def path_preprocess(path: str) -> str:
    # 若已存在该路径，则报错
    if os.path.exists(path):
        raise FileExistsError(f"File {path} already exists.")
    # 创建目录
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

def load_json(file_path: str) -> dict:
    return json.load(open(file_path, 'r'))

def save_json(data: dict, file_path: str) -> str:
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