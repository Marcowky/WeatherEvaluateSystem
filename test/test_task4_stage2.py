import sys

sys.path.append('src')

from evaluation.util import get_geo_stationid_map, get_geo_list

geo_list = get_geo_list()

geo_set = set()

for geo in geo_list:
    if geo in geo_set:
        print(geo)
    geo_set.add(geo)


print(get_geo_stationid_map())