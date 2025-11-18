import sys

sys.path.append('src')

from model.client import ModelClient

client = ModelClient()

res = client.chat_with_prompt_return_text(model="deepseek-ai/DeepSeek-V3", prompt="Hello, how are you? output json", response_format={'type': 'json_object'})

print(res)

from util.multi_thread import run_in_threads

args_list_dict = {
    "model": "deepseek-ai/DeepSeek-V3",
    "prompt": [
        "Hello, how are you?",
        "Hello, how are you?",
        "Hello, how are you?",
        "Hello, how are you?",
    ]
}

results = run_in_threads(client.chat_with_prompt_return_text, args_list_dict, 5)

print(results)