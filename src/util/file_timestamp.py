import os
from datetime import datetime

def get_timestamp() -> str:
    # 返回 YYYYMMDDHHMMSS 格式的时间戳
    return datetime.now().strftime("%Y%m%d%H%M%S")