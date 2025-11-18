import sys

sys.path.append('src')

# from util.data_process import get_geo_division

# print(get_geo_division())

# from evaluation.util import get_geo_list

# print(get_geo_list())
# print(get_geo_list().keys())

from evaluation.util import geo_standardize

print(geo_standardize(["粤北", "惠州北部", "东莞南部", "连南瑶族自治县", "阳山县"]))  # 示例地理位置名称