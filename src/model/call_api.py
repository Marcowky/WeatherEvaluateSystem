from util.multi_thread import run_in_threads

def call_llm_for_data_cleaning_or_analysis(client, model, prompts, max_workers=20):
    args_list_dict = {
        "model": model,
        "prompt": prompts,
        "temperature": 1.0,
        "response_format": {'type': 'json_object'},
    }

    # 多线程向模型发送请求，提升批量效率
    results = run_in_threads(client.chat_with_prompt_return_text, args_list_dict, max_workers=max_workers)

    return results