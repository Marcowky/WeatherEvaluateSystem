import pandas as pd
from model.call_api import call_llm_for_data_cleaning_or_analysis
from prompt.evaluation_prompt import UTIL_PROMPT
from util.data_process import get_geo_division, str_to_json


def get_geo_stationid_map() -> dict[str, list[str]]:
    """获取地理位置名称到站点ID的映射字典"""
    geo_division_df = get_geo_division()
    # 遍历所有列，收集地理位置名称和对应的站点ID
    geo_stationid_map = {}
    for _, row in geo_division_df.iterrows():
        station_id = row['站号+A:K']
        for column in geo_division_df.columns[1:]:
            geo_name = row[column]
            # 若非空
            if pd.notna(geo_name):
                if geo_name not in geo_stationid_map:
                    geo_stationid_map[geo_name] = []
                geo_stationid_map[geo_name].append(station_id)
    return geo_stationid_map


def get_geo_dict_list():
    """获取地理位置名称列表"""
    geo_division_df = get_geo_division()
    # 遍历所有列，收集地理位置名称（跳过第一列）
    std_geo_dict_list = {}
    for column in geo_division_df.columns[1:]:
        std_geo_dict_list[column] = geo_division_df[column].dropna().unique().tolist()
    return std_geo_dict_list


def get_geo_list():
    geo_dict_list = get_geo_dict_list()
    std_geo_list = []
    for _, names in geo_dict_list.items():
        std_geo_list.extend(names)
    return std_geo_list


def get_station_id_set():
    """获取所有站点ID的集合"""
    geo_division_df = get_geo_division()
    station_id_set = set(geo_division_df['站号+A:K'].tolist())
    return station_id_set


def geo_standardize(geo_list: list[str]) -> list[str]:
    """对地理位置名称进行标准化处理"""
    from model.client import ModelClient

    client = ModelClient()
    # 加载标准化的地理位置名称列表和字典
    std_geo_list = get_geo_list()
    std_geo_dict_list = get_geo_dict_list()

    # 去掉多余空格，记录所有未匹配到标准表的地理位置索引
    not_standardized_idx = []
    geo_list = [geo.strip() for geo in geo_list]
    for idx, geo in enumerate(geo_list):
        if geo not in std_geo_list:
            not_standardized_idx.append(idx)
    
    not_standardized_geo = [geo_list[idx] for idx in not_standardized_idx]

    # 对未标准化的地理位置名称进行处理
    # print(f"未标准化的地理位置名称: {not_standardized_geo}")
    llm_standardized_geo = geo_standardize_by_llm(client, std_geo_list, std_geo_dict_list, not_standardized_geo)

    for i, idx in enumerate(not_standardized_idx):
        geo_list[idx] = llm_standardized_geo[i]
    
    return geo_list


def geo_standardize_by_llm(client, std_geo_list, std_geo_dict_list, geo_list: list[str]) -> list[str]:
    """将自然语言描述转化为结构化 JSON"""
    prompt = UTIL_PROMPT.GEO_STANDARDIZE  # 固定提示词模板

    # 初始化映射表，默认加 error_ 前缀表示仍待确认
    geo_map_dict = {geo: f"error_{geo}" for geo in geo_list}
    res_geo_list = geo_list
    
    # 多次尝试，直到成功或达到最大尝试次数
    max_attempts = 5
    attempts = 0
    finish_process = False

    while not finish_process and attempts < max_attempts:
        attempts += 1
        try:
            # 调用模型进行地理位置名称标准化
            orgnized_prompt = prompt.format(geo_list={"ori_geo": res_geo_list}, std_geo_dict_list=std_geo_dict_list)

            response = call_llm_for_data_cleaning_or_analysis(
                client=client,
                model="deepseek-ai/DeepSeek-V3",
                prompt=orgnized_prompt,
            )

            response_json = str_to_json(response)
            llm_std_geo = response_json['std_geo']
            
            # 检查模型规范化的地理位置名称是否在标准列表中
            new_res_geo_list = []
            for ori_geo, std_geo in zip(res_geo_list, llm_std_geo):
                if std_geo in std_geo_list or std_geo == "地区错误":
                    geo_map_dict[ori_geo] = std_geo
                else:
                    new_res_geo_list.append(ori_geo)

            res_geo_list = new_res_geo_list
            if len(res_geo_list) > 0:
                raise ValueError(f"Some geo names are still not standardized: {res_geo_list}.")

            finish_process = True

        except Exception as e:
            print(f"Error during geo standardization: {e}")

    return [geo_map_dict[geo] for geo in geo_list]


def geo_list_to_stationid(geo_list: list[str]) -> list[str]:
    """将地理位置名称转换为对应的站点ID列表"""
    geo_stationid_map = get_geo_stationid_map()
    stationid_set = set()
    for geo in geo_list:
        if geo in geo_stationid_map:
            stationid_set.update(geo_stationid_map[geo])
    # 从小到大排序返回站点ID列表
    stationid_list = sorted(list(stationid_set))
    return stationid_list
