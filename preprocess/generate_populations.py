# coding:utf-8
import pandas as pd
import os
from datetime import datetime, timedelta
from utils.data_util import DataUtil
from utils.compute import TrajectoryToDoorFLow
from utils.topology_graph import IndoorTopoDoorVertex
import random
import copy
from ast import literal_eval
from collections import OrderedDict
import numpy as np

def Timestamp(str_time):

    # str_time = str_time.split('.')[0]
    try:
        return datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S.%f')
    except:
        return datetime.strptime(str_time, '%Y-%m-%d %H:%M:%S')
class TrajectoryRecover(object):
    def __init__(self):
        self.tdf = TrajectoryToDoorFLow()
        self.floor_num = 7
        self.itdv = IndoorTopoDoorVertex(floor_num=self.floor_num)
        self.traj_path = './data/'
        self.filename = 'location20180101.csv'
        self.location_df = self.load_location_data()
        self.interested_columns = ['clientMac', 'x', 'y', 'timeStamp', 'path']
        self.useful_columns = ['clientMac', 'pre_timeStamp', 'pre_floor', 'pre_x_moved',
                               'pre_y_moved', 'pre_region_id', 'pre_point', 'timeStamp', 'floor', 'x_moved',
                               'y_moved', 'region_id', 'point', 'inter_path', 'inter_path_info']
        # clientMac, pre_timeStamp, pre_floor, pre_x_moved, pre_y_moved, pre_region_id, pre_point, timeStamp, floor, x_moved, y_moved, region_id, point, inter_path, inter_path_info

        self.dtypes = {'clientMac': str, 'pre_timeStamp': str, 'pre_floor': int, 'pre_x_moved': float,
                       'pre_y_moved': float, 'pre_region_id': int, 'pre_point': object, 'timeStamp': str, 'floor': int,
                       'x_moved': float, 'y_moved': float, 'region_id': int, 'point': object, 'inter_path': str,
                       'inter_path_info': str}

    def trans_location_df(self, location_df):
        location_df = location_df.loc[location_df['path'].str.contains('/ADSP/HZDS/BLD-B'), :]
        location_df['floor'] = location_df.apply(lambda x: int(x['path'][-1]), axis=1)
        location_df.x = location_df.x.astype(float)
        location_df.y = location_df.y.astype(float)
        location_df['x_moved'] = location_df.apply(
            lambda x: round(DataUtil.transform_point([x['x'], x['y']], x['floor'])[0], 4), axis=1)
        location_df['y_moved'] = location_df.apply(
            lambda x: round(DataUtil.transform_point([x['x'], x['y']], x['floor'])[1], 4), axis=1)
        location_df['region_id'] = location_df.apply(
            lambda x: self.tdf.find_locaton_in_which_region((x['x_moved'], x['y_moved'], x['floor']))[
                0] if self.tdf.find_locaton_in_which_region((x['x_moved'], x['y_moved'], x['floor'])) else 'no', axis=1)
        location_df = location_df[location_df['region_id'] != 'no']

        location_df['timeStamp'] = pd.to_datetime(location_df['timeStamp'])
        location_df['point'] = location_df.apply(lambda x: (x['x_moved'], x['y_moved'], x['floor']), axis=1)
        location_df = location_df.sort_values(by=['timeStamp'], ascending=True)
        location_df = location_df.reset_index(drop=True)
        return location_df

    def load_location_data(self):
        df = pd.read_csv(os.path.join(self.traj_path, self.filename), sep=',', header=0)
        df = df.loc[df['path'].str.contains('/ADSP/HZDS/BLD-1/Floor {floor}'.format(floor=self.floor_num)), :]
        df = df.loc[(df['timeStamp'] >= '2017-01-05 10:00:00') & (df['timeStamp'] <= '2017-01-05 14:00:00'),
             :].reset_index().drop(['index'], axis=1)
        res_df = self.trans_location_df(df)
        return res_df

    def group_by_mac(self):
        group_dfs = self.location_df.groupby(['clientMac'])
        cnt = 0
        for name, group in group_dfs:
            if cnt == 1:
                one_mac_traj = pd.DataFrame(group)
                one_mac_traj = self.trans_location_df(one_mac_traj.loc[:, self.interested_columns])
                break
            cnt += 1

        return one_mac_traj

    def recover_all_macs_traj(self):
        group_dfs = self.location_df.groupby(['clientMac'])
        all_macs_traj = pd.DataFrame()
        cnt = 0
        for name, group in group_dfs:
            # if cnt == 3:
            print('current mac name: %s' % (name))
            if name == '64:9a:be:29:86:60' or name == '0c:1d:af:c4:a6:08' or name == '00:61:71:61:fb:7b' or name == '28:5a:eb:b0:e3:0b':
                continue
            one_mac_traj = pd.DataFrame(group)
            one_mac_traj = self.trans_location_df(one_mac_traj.loc[:, self.interested_columns])
            adjacent_traj_combine = self.recover_one_mac_traj(one_mac_traj)
            if not adjacent_traj_combine.empty:
                adjacent_traj_combine = adjacent_traj_combine.loc[:, self.useful_columns]
                all_macs_traj = pd.concat([all_macs_traj, adjacent_traj_combine], axis=0)
            print('finish recover %d macs' % (cnt), datetime.now())
            # break

            cnt += 1

        all_macs_traj = all_macs_traj.reset_index(drop=True)
        all_macs_traj.to_csv(
            os.path.join(self.traj_path, 'all_macs_traj_floor_{floor}.csv'.format(floor=str(self.floor_num))),
            header=True, index=False)
        print('done recover traj', datetime.now())
        print('all_macs_traj length', len(all_macs_traj))
        print('inter path same', len(all_macs_traj[all_macs_traj['inter_path'] == 'same']))
        print('inter path pending', len(all_macs_traj[all_macs_traj['inter_path'] == 'pending']))
        print('inter path other', len(all_macs_traj[all_macs_traj['inter_path'] == 'adjacent']))

        return all_macs_traj

    def recover_one_mac_traj(self, one_mac_traj):
        # one_mac_traj = self.group_by_mac()
        one_mac_traj_original = copy.copy(one_mac_traj)
        one_mac_traj_shift = one_mac_traj.shift(-1).convert_dtypes()
        one_mac_traj = one_mac_traj.add_prefix('pre_')
        # print(one_mac_traj.loc[samples,:])
        # print(one_mac_traj_shift.loc[0,:])
        adjacent_traj_combine = pd.concat([one_mac_traj, one_mac_traj_shift], axis=1)
        # adjacent_traj_combine.to_csv('./data/adjacent_traj_combine.csv',header=True,index=False)
        # print('len of adjacent_traj_combine',len(adjacent_traj_combine))
        # print(adjacent_traj_combine.iloc[-samples])
        adjacent_traj_combine = adjacent_traj_combine.loc[:len(adjacent_traj_combine) - 2, :]
        # print('len of adjacent_traj_combine later',len(adjacent_traj_combine))
        # print(adjacent_traj_combine.iloc[-samples])
        adjacent_traj_combine = adjacent_traj_combine[(adjacent_traj_combine['pre_floor'] == self.floor_num) & (
                    adjacent_traj_combine['floor'] == self.floor_num)].reset_index(drop=True)
        print('length of one mac traj', len(adjacent_traj_combine))
        if adjacent_traj_combine.empty:
            return pd.DataFrame()
        adjacent_traj_combine['inter_path'] = adjacent_traj_combine.apply(
            lambda x: 'same' if x['pre_region_id'] == x['region_id'] else
            ('adjacent' if self.tdf.door_to_regions_relation_dict.get(self.floor_num).get(
                '-'.join([str(x['pre_region_id']), str(x['region_id'])]), None) else 'pending'), axis=1)
        print('finish completing inter_path type', datetime.now())

        # print('inter_path column',adjacent_traj_combine.loc[:,'inter_path'].unique())
        print('inter path same', len(adjacent_traj_combine[adjacent_traj_combine['inter_path'] == 'same']))
        print('inter path pending', len(adjacent_traj_combine[adjacent_traj_combine['inter_path'] == 'pending']))
        print('inter path adjacent', len(adjacent_traj_combine[adjacent_traj_combine['inter_path'] == 'adjacent']))

        adjacent_traj_combine['inter_path_info'] = adjacent_traj_combine.apply(
            lambda x: {x['region_id']: 1} if x['inter_path'] == 'same'
            else (self.itdv.figure_adjacent_points_path(start_point=x['pre_point'], start_timestamp=x['pre_timeStamp'],
                                                        end_point=x['point'], end_timestamp=x['timeStamp']) if x[
                                                                                                                   'inter_path'] == 'adjacent'
                  else self.itdv.figure_nonadjacent_points_paths(start_point=x['pre_point'],
                                                                 start_timestamp=x['pre_timeStamp'],
                                                                 end_point=x['point'], end_timestamp=x['timeStamp'])),
            axis=1)
        print('finish completing inter_path_info', datetime.now())
        adjacent_traj_combine = adjacent_traj_combine.reset_index(drop=True)
        return adjacent_traj_combine

    def read_recovered_traj_data(self):
        all_macs_traj = pd.read_csv(
            os.path.join(self.traj_path, 'all_macs_traj_floor_{floor}.csv'.format(floor=self.floor_num)), sep=',',
            header=0, dtype=object)
        return all_macs_traj

    def generate_population_snapshot(self, traj_combine_df, shot_time):
        traj_combine_shot_df = traj_combine_df[
            (traj_combine_df['timeStamp'] > shot_time) & (traj_combine_df['pre_timeStamp'] <= shot_time)].reset_index(
            drop=True)
        assert len(traj_combine_shot_df['clientMac']) == len(
            traj_combine_shot_df['clientMac'].unique()), 'mac duplicated in one time shot'
        # print('111',len(traj_combine_shot_df))
        if traj_combine_shot_df.empty:
            return {}
        traj_combine_shot_df['population'] = traj_combine_shot_df.apply(
            lambda x: {x['region_id']: 1} if x['inter_path'] == 'same' else (
                self.figure_population_for_adjacent_points(x['inter_path_info'], shot_time)
                if x['inter_path'] == 'adjacent' else self.figure_population_for_nonadjacent_points(
                    x['inter_path_info'], shot_time)), axis=1)
        population_list = traj_combine_shot_df['population'].values.tolist()
        combined_population_dict = self.combine_multiple_dict(population_list)
        return combined_population_dict

    def generate_flow_snapshot(self, traj_combine, shot_time, time_interval, end_time):
        time_interval = timedelta(seconds=time_interval)
        traj_combine_original = traj_combine[traj_combine['inter_path'] != 'same'].reset_index(drop=True)
        segments_flow_dict = OrderedDict()
        while (shot_time + time_interval) <= end_time:
            traj_combine_df = copy.copy(traj_combine_original)
            segment_time_limit = shot_time + time_interval
            segment_time_cover = (shot_time, segment_time_limit)

            traj_combine_df['shot_position'] = traj_combine_df.apply(
                lambda x: 'traj_in' if x['pre_timeStamp'] >= shot_time and x['timeStamp'] <= segment_time_limit
                else ('left_cross' if x['pre_timeStamp'] < shot_time and x['timeStamp'] > shot_time and x[
                    'timeStamp'] <= segment_time_limit
                      else (
                    'right_cross' if x['pre_timeStamp'] >= shot_time and x['pre_timeStamp'] < segment_time_limit and x[
                        'timeStamp'] > segment_time_limit
                    else ('shot_in' if x['pre_timeStamp'] < shot_time and x['timeStamp'] > segment_time_limit
                          else 'out'))), axis=1)
            traj_combine_df['flow'] = traj_combine_df.apply(lambda x: {} if x['shot_position'] == 'out'
            else (self.figure_flow_p2p(x['inter_path_info'], x['inter_path'], x['pre_timeStamp'], x['timeStamp']) if x[
                                                                                                                         'shot_position'] == 'traj_in'
                  else (self.figure_flow_p2p(x['inter_path_info'], x['inter_path'], shot_time, x['timeStamp']) if x[
                                                                                                                      'shot_position'] == 'left_cross'
                        else (
                self.figure_flow_p2p(x['inter_path_info'], x['inter_path'], x['pre_timeStamp'], segment_time_limit) if
                x['shot_position'] == 'right_cross'
                else self.figure_flow_p2p(x['inter_path_info'], x['inter_path'], shot_time, segment_time_limit)))),
                                                            axis=1)

            segment_flow_list = traj_combine_df['flow'].values.tolist()
            segment_flow_dict = self.combine_multiple_dict(segment_flow_list)
            segments_flow_dict[segment_time_cover] = segment_flow_dict
            shot_time = copy.copy(segment_time_limit)
        return segments_flow_dict

    def figure_flow_p2p(self, inter_path_info, inter_path, start_time, end_time):
        flow_dict = {}
        if inter_path == 'adjacent':
            door_id = inter_path_info['node_id']
            direction = tuple(inter_path_info['direction'])
            timestamp = inter_path_info['timestamp']
            if timestamp >= start_time and timestamp < end_time:
                flow_dict[(door_id, direction)] = 1
        else:
            flow_dict_list = []
            assert inter_path == 'pending', 'inter_path wrong > figure_traj_in_type_flow'
            for path, path_info in inter_path_info.items():
                path_weight = path_info['path_weight']
                path_info_list = path_info['path_info_list']
                path_info_list = list(filter(
                    lambda x: x['timestamp'] >= start_time and x['timestamp'] < end_time and x['node_id'] not in ['s',
                                                                                                                  'e'],
                    path_info_list))
                path_flow_dict_list = list(
                    map(lambda x: {(x['node_id'], tuple(x['direction'])): path_weight}, path_info_list))
                path_flow_combined_dict = self.combine_multiple_dict(path_flow_dict_list)
                flow_dict_list.append(path_flow_combined_dict)
            flow_dict = self.combine_multiple_dict(flow_dict_list)
        return flow_dict

    def figure_population_for_adjacent_points(self, inter_path_info, shot_time):
        direction = inter_path_info['direction']
        start_region = direction[0]
        end_region = direction[1]
        timestamp = inter_path_info['timestamp']
        population_dict = {}
        if shot_time <= timestamp:
            population_dict[start_region] = 1
        else:
            population_dict[end_region] = 1
        return population_dict

    def figure_population_for_nonadjacent_points(self, inter_path_info, shot_time):
        path_population_dict_list = []
        for path, path_info in inter_path_info.items():
            path_weight = path_info['path_weight']
            path_info_list = path_info['path_info_list']
            path_info_list = sorted(path_info_list, key=lambda x: x.get('timestamp'), reverse=False)
            single_path_population_dict = {}
            for node_index, node_info in enumerate(path_info_list):
                if node_info['timestamp'] > shot_time:
                    if node_info['node_id'] == 'e':
                        pre_node = path_info_list[node_index - 1]
                        region = pre_node['direction'][1]
                        single_path_population_dict[region] = path_weight
                        path_population_dict_list.append(single_path_population_dict)
                        break
                    else:
                        region = node_info['direction'][0]
                        single_path_population_dict[region] = path_weight
                        path_population_dict_list.append(single_path_population_dict)
                        break
        population_dict = self.combine_multiple_dict(path_population_dict_list)
        return population_dict

    @staticmethod
    def combine_multiple_dict(dict_list):
        combined_dict = {}
        for each_dict in dict_list:
            for k, v in each_dict.items():
                if k in combined_dict:
                    combined_dict[k] = combined_dict[k] + v
                else:
                    combined_dict[k] = v
        return combined_dict

    def reload_all_combine_traj(self):
        all_macs_traj = self.read_recovered_traj_data()
        all_macs_traj['pre_timeStamp'] = pd.to_datetime(all_macs_traj['pre_timeStamp'])
        all_macs_traj['timeStamp'] = pd.to_datetime(all_macs_traj['timeStamp'])
        all_macs_traj['inter_path_info'] = all_macs_traj['inter_path_info'].apply(lambda x: eval(str(x)))
        all_macs_traj['point'] = all_macs_traj['point'].apply(lambda x: eval(str(x)))
        all_macs_traj['pre_point'] = all_macs_traj['pre_point'].apply(lambda x: eval(str(x)))
        all_macs_traj['pre_region_id'] = all_macs_traj['pre_region_id'].apply(lambda x: int(x))
        all_macs_traj['region_id'] = all_macs_traj['region_id'].apply(lambda x: int(x))
        all_macs_traj['floor'] = all_macs_traj['floor'].apply(lambda x: int(x))
        all_macs_traj['pre_floor'] = all_macs_traj['pre_floor'].apply(lambda x: int(x))

        return all_macs_traj

    @staticmethod
    def extract_bound_points(points):
        x_list = list(map(lambda x: x['x'], points))
        y_list = list(map(lambda x: x['y'], points))
        min_x, max_x = min(x_list), max(x_list)
        min_y, max_y = min(y_list), max(y_list)
        return min_x, min_y, max_x, max_y


    def figure_population_of_regions(self, shot_time):
        all_macs_traj = self.reload_all_combine_traj()
        combined_population_dict = self.generate_population_snapshot(traj_combine_df=all_macs_traj, shot_time=shot_time)
        print('shot_time', shot_time)
        regions = self.itdv.regions
        doors = self.itdv.doors
        regions_pop = []
        region_ids = list(map(lambda x: x['id'], regions))
        for region_id in region_ids:
            if region_id not in combined_population_dict:
                regions_pop.append(0)
            else:
                regions_pop.append(round(combined_population_dict[region_id], 4))
        return region_ids, regions_pop

    def generate_pop_data(self, start_time, time_interval, end_time):
        pop_list = []
        index_list = []
        index_list.append(start_time.strftime('%Y-%m-%d %H:%M:%S'))

        regions_ids, regions_pop = self.figure_population_of_regions(start_time)
        self.figure_physical_adjacency_matrix(regions_ids)
        pop_list.append(regions_pop)
        while start_time + time_interval <= end_time:
            regions_pop_seg = self.figure_population_of_regions(start_time + time_interval)[1]
            pop_list.append(regions_pop_seg)
            index_list.append((start_time + time_interval).strftime('%Y-%m-%d %H:%M:%S'))
            start_time = start_time + time_interval

        assert len(index_list) == len(pop_list), 'index list unequal pop list'

        pop_df = pd.DataFrame(pop_list, index=index_list, columns=regions_ids)
        print('shape of pop_df', pop_df.shape)
        pop_df.to_csv(os.path.join(self.traj_path, 'pop.csv'), header=True, index=True)

    def figure_physical_adjacency_matrix(self, regions_ids):
        print('len of regions_ids', len(regions_ids))
        adj = np.eye(len(regions_ids), len(regions_ids))
        regions = self.itdv.regions
        for i, region_id in enumerate(regions_ids):
            the_region = list(filter(lambda x: x['id'] == region_id, regions))[0]
            neighbor_regions_ids = the_region['connectedRegionsID']
            for neigh_region_id in neighbor_regions_ids:
                neigh_region_id_index = regions_ids.index(neigh_region_id)
                adj[i, neigh_region_id_index] = 1
                adj[neigh_region_id_index, i] = 1

        adj_df = pd.DataFrame(adj.tolist(), index=regions_ids, columns=regions_ids)
        print('shape of adj', adj_df.shape)
        adj_df.to_csv(os.path.join(self.traj_path, 'adj.csv'), index=True, header=True)

shot_time = datetime.strptime('2018-01-01 09:00:00', '%Y-%m-%d %H:%M:%S')
end_time = datetime.strptime('2018-01-31 19:00:00', '%Y-%m-%d %H:%M:%S')
time_interval = 60
tjr = TrajectoryRecover()
recovered_location = tjr.recover_all_macs_traj()
tjr.generate_pop_data(shot_time, time_interval, end_time)

