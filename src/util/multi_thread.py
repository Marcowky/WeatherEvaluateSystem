from concurrent.futures import ThreadPoolExecutor
import traceback
from tqdm import tqdm

def run_in_threads(func, args_list_dict: dict, max_workers: int = 5):
    # 验证每个 arg 的数量
    args_len = 1
    for key, value in args_list_dict.items():
        if isinstance(value, list):
            if len(value) != args_len and args_len != 1:
                raise ValueError("Inconsistent argument lengths in args_list_dict.")
            args_len = len(value)
    
    # 将 arg 添加到 args_list
    args_list = []
    for i in range(args_len):
        cur_arg = {}
        for key, value in args_list_dict.items():
            if isinstance(value, list):
                cur_arg[key] = value[i]
            else:
                cur_arg[key] = value
        args_list.append(cur_arg)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for kwargs in args_list:
            try:
                futures.append(executor.submit(func, **kwargs))
            except Exception as e:
                print(f"Error submitting task with args {kwargs}: {e}")
                traceback.print_exception(type(e), e, e.__traceback__)
                results.append(f"{type(e)}, {str(e)}")

        for future in tqdm(futures, desc="Processing", total=len(futures)):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error during task execution: {e}")
                traceback.print_exception(type(e), e, e.__traceback__)
                results.append(f"{type(e)}, {str(e)}")
    return results