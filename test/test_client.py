import sys

sys.path.append('src')

from model.client import ModelClient

client = ModelClient()

res = client.chat_with_prompt_return_text(model="deepseek-ai/DeepSeek-V3", prompt="Hello, how are you? output json", response_format={'type': 'json_object'})

print(res)