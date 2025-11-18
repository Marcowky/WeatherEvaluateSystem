from util.data_process import load_yaml

def load_config(path: str = "config.yaml") -> dict:
    return load_yaml(path)