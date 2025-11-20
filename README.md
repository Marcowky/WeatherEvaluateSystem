# WeatherEvaluateSystem

> 本 README 由大模型生成，仅供参考！！！

## 项目简介
WeatherEvaluateSystem 是一个围绕气象新闻问答任务（当前以 Task4 为主）的评测工具集。项目聚焦于大模型输出的结构化抽取、地理标准化与温度准确率打分，帮助研发人员以统一标准衡量不同模型的表现。

## 功能亮点
- **多阶段抽取管线**：先利用大模型从原始回答中抽取结构化信息，再自动做地理名称标准化，保证评分的可比性。
- **精细化地理评估**：结合地理区域字典与站点 ID 映射，通过 IoU 与 station-id 交并比衡量空间覆盖。
- **温度区间打分**：提供精确值、四舍五入值与区间评分三套算法，对最高气温及各区域区间进行细粒度评估。
- **数据与脚本解耦**：`data/` 用于存放地理划分、温度观测及样例结果，`src/` 专注于算法逻辑，便于扩展至其他任务。

## 目录速览
```
├── config_example.yaml      # API 配置示例
├── data/                    # 气象原始数据、label 与提示词
├── notebook/                # 探索式分析或实验草稿
├── result/                  # 阶段性评估输出（抽取、标准化、评分等）
├── src/
│   ├── evaluation/          # 评估脚本与指标
│   ├── model/               # OpenAI/SiliconFlow 客户端封装
│   ├── prompt/              # 任务及工具提示词
│   ├── task/                # 任务定义（预留扩展点）
│   └── util/                # 配置、IO、并发等通用工具
├── test/                    # 人工调试脚本与回归用例
└── requirements.txt
```

## 快速上手
1. **准备 Python 环境**  
   建议使用 Python 3.10+，并在仓库根目录创建虚拟环境。
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **安装依赖**  
   ```bash
   pip install -r requirements.txt
   pip install tqdm  # tqdm 在多线程脚本中会用到
   ```

3. **配置 API**  
   复制示例文件并填入自己的 key：
   ```bash
   cp config_example.yaml config.yaml
   ```
   `config.yaml` 的核心结构如下：
   ```yaml
   llm_api:
     siliconflow:
       base_url: https://api.siliconflow.cn/v1
       api_key: "替换为真实 key"
     default: siliconflow
   ```
   如果你需要切换到其他 OpenAI 兼容的服务，只需在 `llm_api` 下新增节点，并将 `default` 指向该节点。

4. **准备数据**  
   - `data/station_info/地理划分_去除空列.csv`：地理层级与站点映射。
   - `data/newspaper/task4/qa_data.json`：Task4 的原始问答与 csv 路径。
   - `data/task4/2024/tmax/*.csv`：根据 `label['input']['csv_data_path']` 指定的文件读取真实温度。  
   所有路径已在脚本中写死，如需要自定义可在运行前重写常量或传参（详见下文）。

## Task4 评估流程
> 建议按顺序执行三个阶段；各脚本均可直接运行，也可以在 notebook/或 REPL 中导入函数进行更灵活的控制。

1. **阶段 1-1：信息抽取**  
   ```bash
   python -m src.evaluation.task4.stage_1_1_info_extract
   ```
   - 输入：`DEFAULT_INPUT_PATH`（默认指向某次模型原始输出 JSON）。  
   - 输出：为每条样本附上 `extracted_info` 字段，文件保存在 `DEFAULT_OUTPUT_PATH`。  
   - 关键点：脚本默认调用 `deepseek-ai/DeepSeek-V3`，并在失败后自动重试，结果写入 `result/evaluation/.../task4_info_extract_by_llm.json`。

2. **阶段 1-2：地理标准化**  
   ```bash
   python -m src.evaluation.task4.stage_1_2_geo_standardize
   ```
   - 输入：阶段 1-1 的输出。  
   - 输出：`specific_regions` 和 `max_temp` 会新增 `std_geo` 字段，确保后续能和标准答案按站点对齐。  
   - 实现：调用 `evaluation.util.geo_standardize`，必要时会向 LLM 请求纠错/映射。

3. **阶段 2：准确率计算**  
   ```bash
   python -m src.evaluation.task4.stage_2_scoring
   ```
   - 输入：阶段 1-2 的 JSON + `data/newspaper/task4/qa_data_info_extract_geo_standardize.json` 作为标准答案。  
   - 输出：  
     - 带有 `accuracy_score` 字段的完整结果（地理 IoU + 温度区间打分）。  
     - `_summary.json` 汇总平均分，便于快速比较模型。

### 路径与自定义
- 三个脚本顶部的 `DEFAULT_INPUT_PATH/OUTPUT_PATH` 等常量可按需修改。  
- 若希望在不改源码的情况下自定义，可在其他 Python 脚本中导入函数，例如：
  ```python
  from src.evaluation.task4 import stage_1_1_info_extract as s11
  custom_output = s11.info_extract_by_llm(my_model_result)
  ```
- 多线程参数（默认 `max_workers=5`）可在 `util.multi_thread.run_in_threads` 中调整。

## 测试与调试
- 项目使用 `pytest`（待补充正式用例），目前 `test/` 目录下的脚本主要是人工检验流程的示例。  
- 推荐在提交前至少手动跑通关键脚本，或在 notebook 中抽样检查 `extracted_info/std_geo/accuracy_score` 的结构。  
- 若需要构建自动化测试，可以 `test/test_task4_stage2.py` 为蓝本，编写针对特定数据集的回归测试。

## 开发指南
1. **模块化扩展**  
   - 评估逻辑集中在 `src/evaluation`，新增任务时优先复制现有的 Task4 结构（阶段化脚本 + metrics + util）。  
   - Prompt 模板统一保存在 `src/prompt`，修改时请保持类常量形式，避免硬编码在业务脚本中。

2. **配置与密钥**  
   - 所有与 LLM 连接相关的配置必须通过 `config.yaml` 加载，不要将密钥写进代码或数据文件。  
   - 若需要支持多个供应商，实现对应的 `llm_api.xxx` 节点即可，`ModelClient` 会根据 `default` 自动选择。

3. **数据管理**  
   - 大文件默认放在 `data/`，提交前确认是否可公开；若涉及隐私数据，请改用占位符或说明获取方式。  
   - 新增的中间结果请写入 `result/` 并使用 `util.data_process.path_preprocess`，以 timestamp 避免覆盖。

4. **编码与样式**  
   - 使用 `typing` 注解和简短注释解释复杂逻辑，例如多线程、IoU 配对、异常重试。  
   - 涉及并发的函数尽量保持纯函数式输入输出，避免共享状态造成竞态。

5. **提交前检查**  
   - 格式化/静态检查目前未强制，可根据需要引入 `ruff`、`black` 等工具。  
   - 至少确保关键脚本（阶段 1/2）能在示例数据上跑通，并且不会因为路径/配置缺失抛异常。

## 常见问题
- **Rate limit/429**：`ModelClient.chat_with_messages` 已内置指数重试，可根据需要调整 `PENDING_SECOND`、`MAX_ATTEMPTS`。  
- **路径不存在**：脚本中默认路径遵循本仓库结构，若将项目移动到其他位置，请同步修改 `DEFAULT_*` 常量或使用绝对路径。
