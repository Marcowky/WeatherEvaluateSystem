import sys

sys.path.append('src')

from prompt.evaluation_prompt import task4_prompt

print(task4_prompt.EXTRACT_INFO.format(original_text="你的原始文本"))