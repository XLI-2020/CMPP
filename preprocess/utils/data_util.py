
import collections
import math
import numpy as np
from datetime import timedelta




class DataUtil(object):

    __scale_dict__ = {
        1: 25.5,
        2: 25.5,
        3: 25.5,
        4: 28,
        5: 57.5,
        6: 25.7,
        7: 25,
    }

    __rotAng_dict__ = {
        1: 3.3,
        2: 3.3,
        3: 3.3,
        4: 3.3,
        5: 3.3,
        6: 3.3,
        7: 3.3,
    }

    __diffX_dict__ = {
        1: 2290,
        2: 2290,
        3: 2230,
        4: 2230,
        5: 2230,
        6: 2230,
        7: 2250,
    }

    __diffY_dict__ = {
        1: 550,
        2: 700,
        3: 800,
        4: 700,
        5: 730,
        6: 560,
        7: 570,
    }

    __map_parser_dict__ = {
        # samples: MapParser(samples, 'map_data/'),
        # 2: MapParser(2, 'map_data/'),
        # 3: MapParser(3, 'map_data/'),
        # 4: MapParser(4, 'map_data/'),
        # # 5: MapParser(5, 'map_data/'),
        # 6: MapParser(6, 'map_data/'),
        # 7: MapParser(7, 'map_data/'),
    }

    @staticmethod
    def split_to_sub_sequence(data_dict):

        keys = data_dict.keys()
        values = data_dict.values()

        segs = collections.OrderedDict()
        tmp = collections.OrderedDict()
        pre_value = None
        for idx in range(0, len(keys)):
            cur_value = values[idx]
            if pre_value is None:
                pre_value = cur_value
                tmp[keys[idx]] = cur_value
            else:
                if pre_value == cur_value:
                    pre_value = cur_value
                    tmp[keys[idx]] = cur_value
                else:
                    segs[(tmp.keys()[0], tmp.keys()[-1])] = tmp
                    tmp = collections.OrderedDict()
                    pre_value = cur_value
                    tmp[keys[idx]] = cur_value
        segs[(tmp.keys()[0], tmp.keys()[-1])] = tmp

        return segs

    @staticmethod
    def add_data(data_dict, v):
        for items in v.items():
            data_dict[items[0]] = items[1]

    @staticmethod
    def merge_value(smoothed_data_dict, smoothed_value, subsequence):
        new_subsequence = collections.OrderedDict()
        for kk in subsequence.keys():
            new_subsequence[kk] = smoothed_value
        DataUtil.add_data(smoothed_data_dict, new_subsequence)

    @staticmethod
    def merge_unstable_sequence(tmp_cluster):

        if len(tmp_cluster.keys()) == 1:
            return tmp_cluster

        merged_cluster = collections.OrderedDict()
        keys = tmp_cluster.keys()
        values = tmp_cluster.values()
        profile = collections.OrderedDict()
        merge_value = None
        idx_start = 0
        idx_end = 0
        for idx in range(0, len(keys)):
            cur_duration = (values[idx].keys()[-1] - values[idx].keys()[0])
            cur_value = list(set(values[idx].values()))[0]
            # print cur_duration, cur_value, len(values[idx])
            if merge_value is None:
                profile[cur_value] = cur_duration.seconds
                merge_value = cur_value
                idx_start = idx
                idx_end = idx + 1
            else:
                if abs(merge_value - cur_value) <= 2:
                    if cur_value in profile.keys():
                        tmp = profile[cur_value]
                        tmp += cur_duration.seconds
                        profile[cur_value] = tmp
                    else:
                        profile[cur_value] = cur_duration.seconds
                    merge_value = DataUtil.find_merge_value(profile)
                    idx_end = idx + 1
                else:
                    merged_data = DataUtil.to_merge_value(idx_start, idx_end, values, merge_value)
                    merged_cluster[merged_data.keys()[0], merged_data.keys()[-1]] = merged_data
                    profile = collections.OrderedDict()
                    profile[cur_value] = cur_duration.seconds
                    merge_value = cur_value
                    idx_start = idx
                    idx_end = idx + 1
                if idx == len(keys) - 1:
                    merged_data = DataUtil.to_merge_value(idx_start, idx_end, values, merge_value)
                    merged_cluster[merged_data.keys()[0], merged_data.keys()[-1]] = merged_data
        return merged_cluster

    @staticmethod
    def merge_stable_sequence(tmp_cluster, mode='against', ratio_threshold=0.33):

        if len(tmp_cluster) >= 4:
            print('Wrong parameter [tmp_cluster]')

        if len(tmp_cluster) == 1:
            new_tmp_cluster = collections.OrderedDict()
            for item in tmp_cluster:
                new_tmp_cluster[item[0]] = item[1]
            return new_tmp_cluster

        if len(tmp_cluster) == 2:
            v_1 = tmp_cluster[0][1]
            v_2 = tmp_cluster[1][1]
            duration_1 = v_1.keys()[-1] - v_1.keys()[0]
            duration_2 = v_2.keys()[-1] - v_2.keys()[0]
            if duration_2.seconds != 0:
                ratio = float(duration_1.seconds) / duration_2.seconds
            else:
                ratio = 0

            if ratio > 1:
                ratio = float(1) / ratio

            new_tmp_cluster = collections.OrderedDict()
            for item in tmp_cluster:
                new_tmp_cluster[item[0]] = item[1]

            if ratio < ratio_threshold:
                return DataUtil.merge_unstable_sequence(new_tmp_cluster)
            else:
                return new_tmp_cluster

        if (len(tmp_cluster) == 3) and (mode == 'combine'):
            new_tmp_cluster = collections.OrderedDict()
            for item in tmp_cluster:
                new_tmp_cluster[item[0]] = item[1]
            return DataUtil.merge_unstable_sequence(new_tmp_cluster)

        if (len(tmp_cluster) == 3) and (mode == 'against'):
            merged_cluster = collections.OrderedDict()
            stable_head = tmp_cluster[0]
            stable_tail = tmp_cluster[-1]
            stable_cur = tmp_cluster[1]

            value_head = list(set(stable_head[1].values()))[0]
            value_tail = list(set(stable_tail[1].values()))[0]
            value_cur = list(set(stable_cur[1].values()))[0]

            if abs(value_cur - value_head) > abs(value_cur - value_tail):
                merged_cluster[stable_head[0]] = stable_head[1]
                new_tmp_cluster = collections.OrderedDict()
                new_tmp_cluster[stable_cur[0]] = stable_cur[1]
                new_tmp_cluster[stable_tail[0]] = stable_tail[1]
                sub_merged_cluster = DataUtil.merge_unstable_sequence(new_tmp_cluster)
                for (kk, vv) in sub_merged_cluster.items():
                    merged_cluster[kk] = vv
            elif abs(value_cur - value_head) < abs(value_cur - value_tail):
                new_tmp_cluster = collections.OrderedDict()
                new_tmp_cluster[stable_head[0]] = stable_head[1]
                new_tmp_cluster[stable_cur[0]] = stable_cur[1]
                sub_merged_cluster = DataUtil.merge_unstable_sequence(new_tmp_cluster)
                for (kk, vv) in sub_merged_cluster.items():
                    merged_cluster[kk] = vv
                merged_cluster[stable_tail[0]] = stable_tail[1]
            else:
                if value_head == value_tail:
                    new_tmp_cluster = collections.OrderedDict()
                    new_tmp_cluster[stable_head[0]] = stable_head[1]
                    new_tmp_cluster[stable_cur[0]] = stable_cur[1]
                    new_tmp_cluster[stable_tail[0]] = stable_tail[1]
                    sub_merged_cluster = DataUtil.merge_unstable_sequence(new_tmp_cluster)
                    for (kk, vv) in sub_merged_cluster.items():
                        merged_cluster[kk] = vv
                else:
                    duration_head = stable_head[0][1] - stable_head[0][0]
                    duration_tail = stable_tail[0][1] - stable_tail[0][0]
                    # print duration_head, duration_tail
                    if duration_head.seconds < duration_tail.seconds:
                        merged_cluster[stable_head[0]] = stable_head[1]
                        new_tmp_cluster = collections.OrderedDict()
                        new_tmp_cluster[stable_cur[0]] = stable_cur[1]
                        new_tmp_cluster[stable_tail[0]] = stable_tail[1]
                        sub_merged_cluster = DataUtil.merge_unstable_sequence(new_tmp_cluster)
                        for (kk, vv) in sub_merged_cluster.items():
                            merged_cluster[kk] = vv
                    else:
                        new_tmp_cluster = collections.OrderedDict()
                        new_tmp_cluster[stable_head[0]] = stable_head[1]
                        new_tmp_cluster[stable_cur[0]] = stable_cur[1]
                        sub_merged_cluster = DataUtil.merge_unstable_sequence(new_tmp_cluster)
                        for (kk, vv) in sub_merged_cluster.items():
                            merged_cluster[kk] = vv
                        merged_cluster[stable_tail[0]] = stable_tail[1]

            # for (k, v) in merged_cluster.items():
            #     print k, k[samples] - k[0], v.values()[0]

            return merged_cluster

    @staticmethod
    def find_merge_value(profile):
        sorted_by_value_list = sorted(profile.items(), key=lambda d: d[1], reverse=True)
        return sorted_by_value_list[0][0]

    @staticmethod
    def to_merge_value(idx_start, idx_end, values, merge_value):
        merged_data = collections.OrderedDict()
        for idx in range(idx_start, idx_end):
            cur_data = values[idx]
            for kk in cur_data.keys():
                merged_data[kk] = merge_value
        return merged_data

    @staticmethod
    def merge_score(duration_1, duration_2, value_1, value_2):
        print(duration_1, duration_2, value_1, value_2)
        if duration_1 >= duration_2:
            duration_ratio = float(duration_2.seconds) / duration_1.seconds
        else:
            duration_ratio = float(duration_1.seconds) / duration_2.seconds
        return abs(value_1 - value_2), duration_ratio
        # return math.exp(-abs(value_1 - value_2)) * 10 + duration_ratio

    @staticmethod
    def format_mac_address(client_mac):
        formatted_str = ''
        items = client_mac.split(':')
        for item in items:
            formatted_str += item
        return formatted_str

    @staticmethod
    def euclidean_dist_points(p_1, p_2):
        x_1 = p_1[0]
        x_2 = p_2[0]
        y_1 = p_1[1]
        y_2 = p_2[1]
        return DataUtil.euclidean_dist(x_1, y_1, x_2, y_2)

    @staticmethod
    def euclidean_dist(x_1, y_1, x_2, y_2):
        return math.sqrt((x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2))

    @staticmethod
    def transform_point(point, floor_value):
        x = point[0]
        y = point[1]
        x_scaled = - x * DataUtil.__scale_dict__[floor_value]
        y_scaled = y * DataUtil.__scale_dict__[floor_value]

        radians = float(DataUtil.__rotAng_dict__[floor_value]) * math.pi / 180
        x_rotated = x_scaled * math.cos(radians) - y_scaled * math.sin(radians)
        y_rotated = x_scaled * math.sin(radians) + y_scaled * math.cos(radians)

        x_moved = x_rotated + DataUtil.__diffX_dict__[floor_value]
        y_moved = y_rotated + DataUtil.__diffY_dict__[floor_value]
        new_point = (x_moved, y_moved)
        return new_point

    @staticmethod
    def compute_variance(data_list):
        floor_value = data_list[0][2]
        narray = np.array(data_list)[:, :2] / DataUtil.__scale_dict__[floor_value]
        # print narray
        mean = sum(narray) / len(narray)
        diff = narray - mean
        list = [math.sqrt(item) for item in np.sum((diff * diff), axis=1)]
        # print sum(list) / len(list), sum(list), list
        return sum(list) / len(list)

    @staticmethod
    def compute_travel_distance_turns(data_list):
        floor_value = data_list[0][2]
        narray = np.array(data_list)[:, :2] / DataUtil.__scale_dict__[floor_value]
        travel_distance = 0
        turns = 0
        pre_direction = None
        for i in range(0, len(narray) - 1):
            pt1 = (narray[i][0], narray[i][1])
            pt2 = (narray[i + 1][0], narray[i + 1][1])
            travel_distance += DataUtil.euclidean_dist_points(pt1, pt2)
            cur_direction = math.atan2(float(pt1[1] - pt2[1]), float(pt1[0] - pt2[0])) * 180.0 / math.pi
            if pre_direction is None:
                pre_direction = cur_direction
            else:
                if abs(pre_direction - cur_direction) >= 90:
                    turns += 1
                    pre_direction = cur_direction
        # print travel_distance, turns, len(data_list)
        return travel_distance, turns

    @staticmethod
    def get_centroid_pt(data_list):
        narray = np.array([(item[1][0], item[1][1]) for item in data_list])
        centroid_array = np.sum(narray, axis=0) / len(narray)
        centroid_pt = (centroid_array[0], centroid_array[1])
        return centroid_pt

    @staticmethod
    def get_spatial_temporal_range_overlap(pre_data, cur_data):

        floor_value = cur_data[0][1][2]
        p_t_s = pre_data[0][0]
        p_t_e = pre_data[-1][0]
        c_t_s = cur_data[0][0]
        c_t_e = cur_data[-1][0]

        if (p_t_s - c_t_s > timedelta(seconds=0)) and (p_t_s - c_t_e <= timedelta(seconds=0)):
            return True
        if (p_t_e - c_t_s > timedelta(seconds=0)) and (p_t_e - c_t_e <= timedelta(seconds=0)):
            return True
        if (c_t_s - p_t_s > timedelta(seconds=0)) and (c_t_s - p_t_e <= timedelta(seconds=0)):
            return True
        if (c_t_e - p_t_s > timedelta(seconds=0)) and (c_t_e - p_t_e <= timedelta(seconds=0)):
            return True

        p_mbr = DataUtil.get_mbr(pre_data)
        c_mbr = DataUtil.get_mbr(cur_data)
        # print p_mbr, len(pre_data)
        # print c_mbr, len(cur_data)

        intersection_area = DataUtil.compute_area(
            DataUtil.expand_mbr(p_mbr, 0.5 * DataUtil.__scale_dict__[floor_value]), c_mbr)
        if intersection_area > 0:
            return True

        return False

    @staticmethod
    def get_mbr(data):
        p_min_x = p_min_y = float('inf')
        p_max_x = p_max_y = 0
        for (t, l) in data:
            x = l[0]
            y = l[1]
            if x < p_min_x:
                p_min_x = x
            if x > p_max_x:
                p_max_x = x
            if y < p_min_y:
                p_min_y = y
            if y > p_max_y:
                p_max_y = y
        mbr = (p_min_x, p_min_y, p_max_x, p_max_y)
        return mbr

    @staticmethod
    def compute_area(mbr1, mbr2):
        a = mbr1[0]
        b = mbr1[1]
        c = mbr1[2]
        d = mbr1[3]
        e = mbr2[0]
        f = mbr2[1]
        g = mbr2[2]
        h = mbr2[3]
        if max(a, e) > min(c, g):
            x = 0
        else:
            x = min(c, g) - max(a, e)
        if max(b, f) > min(d, h):
            y = 0
        else:
            y = min(d, h) - max(b, f)
        return x * y
    @staticmethod
    def expand_mbr(mbr, r):
        a = mbr[0]
        b = mbr[1]
        c = mbr[2]
        d = mbr[3]
        expanded_mbr = (a - r, b - r, c + r, d + r)
        return expanded_mbr

