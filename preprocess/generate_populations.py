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
        self.vldb_data_path = './data/vldb_data'
        self.filename = 'location20170105.csv'
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
        print('0 len of df', len(df))
        df = df.loc[df['path'].str.contains('/ADSP/HZDS/BLD-1/Floor {floor}'.format(floor=self.floor_num)), :]
        df = df.loc[(df['timeStamp'] >= '2017-01-05 10:00:00') & (df['timeStamp'] <= '2017-01-05 14:00:00'),
             :].reset_index().drop(['index'], axis=1)
        print('1 len of df', len(df))
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
        '''
        dtypes {'pre_clientMac': dtype('O'), 'pre_x': dtype('float64'), 'pre_y': dtype('float64'), 'pre_timeStamp': dtype('<M8[ns]'), 'pre_path': dtype('O'), 'pre_floor': dtype('int64'), 'pre_x_moved': dtype('float64'), 'pre_y_moved': dtype('float64'), 'pre_region_id': dtype('int64'), 'pre_point': dtype('O'), 'clientMac': StringDtype, 'x': dtype('float64'), 'y': dtype('float64'), 'timeStamp': dtype('<M8[ns]'), 'path': StringDtype, 'floor': Int64Dtype(), 'x_moved': dtype('float64'), 'y_moved': dtype('float64'), 'region_id': Int64Dtype(), 'point': dtype('O'), 'inter_path': dtype('O'), 'inter_path_info': dtype('O')}
        columns Index(['pre_clientMac', 'pre_x', 'pre_y', 'pre_timeStamp', 'pre_path',
       'pre_floor', 'pre_x_moved', 'pre_y_moved', 'pre_region_id', 'pre_point',
       'clientMac', 'x', 'y', 'timeStamp', 'path', 'floor', 'x_moved',
       'y_moved', 'region_id', 'point', 'inter_path', 'inter_path_info'],
      dtype='object')

        :return:
        '''
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
        """
        doorID x y endtime, partition1-partition2, flow, partition2-partition1, flow; endtime, partition.....
        :return:
        """
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
        """
        {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:51.612426'), 'direction': [286, 287]}
        {('s', 497, 498, 500, 'e'): {'path_weight': 0.0389, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:22.750395'), 'direction': [293, 286]}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:24.993488'), 'direction': [286, 159]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:26.187184'), 'direction': [159, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 497, 500, 498, 'e'): {'path_weight': 0.0273, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:22.227774'), 'direction': [293, 287]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:23.701956'), 'direction': [287, 159]}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:24.539246'), 'direction': [159, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 497, 500, 'e'): {'path_weight': 0.05, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:23.251359'), 'direction': [293, 287]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:25.954555'), 'direction': [287, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 497, 455, 'e'): {'path_weight': 0.0606, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:23.728412'), 'direction': [293, 287]}, {'node_id': 455, 'timestamp': Timestamp('2017-01-05 12:49:24.795107'), 'direction': [287, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 497, 462, 'e'): {'path_weight': 0.031, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:22.394613'), 'direction': [293, 287]}, {'node_id': 462, 'timestamp': Timestamp('2017-01-05 12:49:24.203017'), 'direction': [287, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 497, 'e'): {'path_weight': 0.061, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:23.749958'), 'direction': [293, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 498, 497, 500, 'e'): {'path_weight': 0.0279, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:23.300413'), 'direction': [293, 286]}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:24.909401'), 'direction': [286, 287]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:26.416960'), 'direction': [287, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 498, 497, 455, 'e'): {'path_weight': 0.0309, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:23.548965'), 'direction': [293, 286]}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:25.331799'), 'direction': [286, 287]}, {'node_id': 455, 'timestamp': Timestamp('2017-01-05 12:49:25.875712'), 'direction': [287, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 498, 497, 462, 'e'): {'path_weight': 0.0208, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:22.713394'), 'direction': [293, 286]}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:23.911801'), 'direction': [286, 287]}, {'node_id': 462, 'timestamp': Timestamp('2017-01-05 12:49:25.124449'), 'direction': [287, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 498, 500, 497, 'e'): {'path_weight': 0.0272, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:23.244992'), 'direction': [293, 159]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:24.080611'), 'direction': [159, 287]}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:25.551851'), 'direction': [287, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 498, 500, 455, 'e'): {'path_weight': 0.0291, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:23.404980'), 'direction': [293, 159]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:24.300149'), 'direction': [159, 287]}, {'node_id': 455, 'timestamp': Timestamp('2017-01-05 12:49:25.939220'), 'direction': [287, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 498, 500, 462, 'e'): {'path_weight': 0.0191, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:22.573409'), 'direction': [293, 159]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:23.159055'), 'direction': [159, 287]}, {'node_id': 462, 'timestamp': Timestamp('2017-01-05 12:49:25.277681'), 'direction': [287, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 498, 500, 'e'): {'path_weight': 0.0447, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:24.690798'), 'direction': [293, 159]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:26.064567'), 'direction': [159, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 498, 457, 455, 'e'): {'path_weight': 0.0252, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:23.076515'), 'direction': [293, 159]}, {'node_id': 457, 'timestamp': Timestamp('2017-01-05 12:49:24.187442'), 'direction': [159, 177]}, {'node_id': 455, 'timestamp': Timestamp('2017-01-05 12:49:26.084099'), 'direction': [177, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 498, 492, 490, 'e'): {'path_weight': 0.0231, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:22.905702'), 'direction': [293, 159]}, {'node_id': 492, 'timestamp': Timestamp('2017-01-05 12:49:24.339451'), 'direction': [159, 284]}, {'node_id': 490, 'timestamp': Timestamp('2017-01-05 12:49:25.344003'), 'direction': [284, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 498, 'e'): {'path_weight': 0.0347, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:23.865438'), 'direction': [293, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 490, 492, 498, 'e'): {'path_weight': 0.023, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 490, 'timestamp': Timestamp('2017-01-05 12:49:22.498480'), 'direction': [293, 284]}, {'node_id': 492, 'timestamp': Timestamp('2017-01-05 12:49:23.498264'), 'direction': [284, 159]}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:24.925208'), 'direction': [159, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 490, 492, 500, 'e'): {'path_weight': 0.027, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 490, 'timestamp': Timestamp('2017-01-05 12:49:22.758821'), 'direction': [293, 284]}, {'node_id': 492, 'timestamp': Timestamp('2017-01-05 12:49:23.932304'), 'direction': [284, 159]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:26.435776'), 'direction': [159, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 490, 497, 500, 'e'): {'path_weight': 0.0304, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 490, 'timestamp': Timestamp('2017-01-05 12:49:22.982236'), 'direction': [293, 286]}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:24.719879'), 'direction': [286, 287]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:26.364106'), 'direction': [287, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 490, 497, 455, 'e'): {'path_weight': 0.034, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 490, 'timestamp': Timestamp('2017-01-05 12:49:23.218136'), 'direction': [293, 286]}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:25.162571'), 'direction': [286, 287]}, {'node_id': 455, 'timestamp': Timestamp('2017-01-05 12:49:25.761661'), 'direction': [287, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 490, 497, 462, 'e'): {'path_weight': 0.0221, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 490, 'timestamp': Timestamp('2017-01-05 12:49:22.443027'), 'direction': [293, 286]}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:23.707995'), 'direction': [286, 287]}, {'node_id': 462, 'timestamp': Timestamp('2017-01-05 12:49:25.000671'), 'direction': [287, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 490, 498, 500, 'e'): {'path_weight': 0.0415, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 490, 'timestamp': Timestamp('2017-01-05 12:49:23.706768'), 'direction': [293, 286]}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:24.856466'), 'direction': [286, 159]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:26.131677'), 'direction': [159, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 490, 'e'): {'path_weight': 0.0438, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 490, 'timestamp': Timestamp('2017-01-05 12:49:23.857317'), 'direction': [293, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 134, 137, 462, 'e'): {'path_weight': 0.0326, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 134, 'timestamp': Timestamp('2017-01-05 12:49:21.503551'), 'direction': [293, 282]}, {'node_id': 137, 'timestamp': Timestamp('2017-01-05 12:49:23.411153'), 'direction': [282, 178]}, {'node_id': 462, 'timestamp': Timestamp('2017-01-05 12:49:24.053109'), 'direction': [178, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}, ('s', 134, 'e'): {'path_weight': 0.1943, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}, {'node_id': 134, 'timestamp': Timestamp('2017-01-05 12:49:24'), 'direction': [293, 286]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:27'), 'direction': None}]}}
        :param inter_path_info:
        :param inter_path:
        :param start_time:
        :param end_time:
        :return:
        """
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
        '''
        {('s', 497, 500, 'e'): {'path_weight': 0.1271, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:16'), 'direction': None}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:17.876132'), 'direction': [286, 287]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:20.128796'), 'direction': [287, 293]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}]}, ('s', 497, 455, 'e'): {'path_weight': 0.1541, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:16'), 'direction': None}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:18.273676'), 'direction': [286, 287]}, {'node_id': 455, 'timestamp': Timestamp('2017-01-05 12:49:19.162588'), 'direction': [287, 293]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}]}, ('s', 497, 462, 'e'): {'path_weight': 0.0788, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:16'), 'direction': None}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:17.162177'), 'direction': [286, 287]}, {'node_id': 462, 'timestamp': Timestamp('2017-01-05 12:49:18.669180'), 'direction': [287, 293]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}]}, ('s', 497, 'e'): {'path_weight': 0.195, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:16'), 'direction': None}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:18.877236'), 'direction': [286, 293]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}]}, ('s', 498, 500, 497, 'e'): {'path_weight': 0.0761, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:16'), 'direction': None}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:18.057411'), 'direction': [286, 159]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:18.823209'), 'direction': [159, 287]}, {'node_id': 497, 'timestamp': Timestamp('2017-01-05 12:49:20.171519'), 'direction': [287, 293]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}]}, ('s', 498, 500, 455, 'e'): {'path_weight': 0.0741, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:16'), 'direction': None}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:18.004150'), 'direction': [286, 159]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:18.750124'), 'direction': [159, 287]}, {'node_id': 455, 'timestamp': Timestamp('2017-01-05 12:49:20.116017'), 'direction': [287, 293]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}]}, ('s', 498, 500, 462, 'e'): {'path_weight': 0.0485, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:16'), 'direction': None}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:17.311174'), 'direction': [286, 159]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:17.799212'), 'direction': [159, 287]}, {'node_id': 462, 'timestamp': Timestamp('2017-01-05 12:49:19.564733'), 'direction': [287, 293]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}]}, ('s', 498, 500, 'e'): {'path_weight': 0.1138, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:16'), 'direction': None}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:19.075665'), 'direction': [286, 159]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:20.220472'), 'direction': [159, 293]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}]}, ('s', 498, 457, 455, 'e'): {'path_weight': 0.064, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:16'), 'direction': None}, {'node_id': 498, 'timestamp': Timestamp('2017-01-05 12:49:17.730429'), 'direction': [286, 159]}, {'node_id': 457, 'timestamp': Timestamp('2017-01-05 12:49:18.656202'), 'direction': [159, 177]}, {'node_id': 455, 'timestamp': Timestamp('2017-01-05 12:49:20.236749'), 'direction': [177, 293]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}]}, ('s', 490, 492, 500, 'e'): {'path_weight': 0.0686, 'path_info_list': [{'node_id': 's', 'timestamp': Timestamp('2017-01-05 12:49:16'), 'direction': None}, {'node_id': 490, 'timestamp': Timestamp('2017-01-05 12:49:17.465684'), 'direction': [286, 284]}, {'node_id': 492, 'timestamp': Timestamp('2017-01-05 12:49:18.443587'), 'direction': [284, 159]}, {'node_id': 500, 'timestamp': Timestamp('2017-01-05 12:49:20.529813'), 'direction': [159, 293]}, {'node_id': 'e', 'timestamp': Timestamp('2017-01-05 12:49:21'), 'direction': None}]}}
        :param inter_path_info:
        :param shot_time:
        :return:
        '''
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

    def reshape_population_flow_data(self, shot_time, time_interval, end_time):
        all_macs_traj = self.reload_all_combine_traj()
        combined_population_dict = self.generate_population_snapshot(traj_combine_df=all_macs_traj, shot_time=shot_time)
        segments_flow_dict = self.generate_flow_snapshot(traj_combine=all_macs_traj, shot_time=shot_time,
                                                         time_interval=time_interval, end_time=end_time)
        print('combined_population_dict:\n', combined_population_dict)
        # print('segments_flow_dict:\n',segments_flow_dict)
        regions = self.itdv.regions
        doors = self.itdv.doors
        region_ids = list(map(lambda x: [x['id'], self.extract_bound_points(x['points'])], regions))
        # make up the missing zero population regions
        for region in region_ids:
            region_id = int(region[0])
            x1 = round(region[1][0], 4)
            y1 = round(region[1][1], 4)
            x2 = round(region[1][2], 4)
            y2 = round(region[1][3], 4)
            if region_id not in combined_population_dict:
                combined_population_dict[region_id] = [x1, y1, x2, y2, 0.0001]
            else:
                combined_population_dict[region_id] = [x1, y1, x2, y2, round(combined_population_dict[region_id], 4)]

        population_df = pd.DataFrame(combined_population_dict, index=['x1', 'y1', 'x2', 'y2', 'population']).T
        population_df.to_csv(
            os.path.join(self.vldb_data_path, 'population_shot_floor_{floor}.txt'.format(floor=self.floor_num)),
            index=True, header=False, sep='\t')

        doors_info = list(map(lambda x: (
        x['id'], ((x['line']['x1'] + x['line']['x2']) / 2, (x['line']['y1'] + x['line']['y2']) / 2),
        x['connectedRegionsID']), doors))
        normal_doors_info = list(filter(lambda x: len(x[2]) == 2, doors_info))
        outlier_door_info = list(filter(lambda x: len(x[2]) != 2, doors_info))
        print('outlier_door_info', outlier_door_info)

        flow_shot_dict = {}
        for door in normal_doors_info:
            door_id = int(door[0])
            door_x = round(door[1][0], 4)
            door_y = round(door[1][1], 4)
            door_pos_direction = tuple(door[2])
            door_neg_direction = tuple([door[2][1], door[2][0]])
            door_pos_tuple = (door_id, door_pos_direction)
            door_neg_tuple = (door_id, door_neg_direction)
            door_flow_slot_info_list = []
            for time_span, flow_info_dict in segments_flow_dict.items():
                end_time = time_span[1]
                end_time_delta = (end_time - shot_time).total_seconds()
                if door_pos_tuple in flow_info_dict:
                    door_pos_flow = round(flow_info_dict[door_pos_tuple], 4)
                else:
                    door_pos_flow = 0
                if door_neg_tuple in flow_info_dict:
                    door_neg_flow = round(flow_info_dict[door_neg_tuple], 4)
                else:
                    door_neg_flow = 0
                door_flow_str = ','.join(
                    [str(end_time_delta), '-'.join([str(door_pos_direction[0]), str(door_pos_direction[1])]),
                     str(door_pos_flow),
                     '-'.join([str(door_neg_direction[0]), str(door_neg_direction[1])]), str(door_neg_flow)])
                door_flow_slot_info_list.append(door_flow_str)

            door_flow_slot_info_str = ';'.join(door_flow_slot_info_list)
            flow_shot_dict[door_id] = [door_x, door_y, door_flow_slot_info_str]
        flow_shot_df = pd.DataFrame(flow_shot_dict, index=['x', 'y', 'flow']).T

        flow_shot_df.to_csv(
            os.path.join(self.vldb_data_path, 'flow_shot_floor_{floor}.txt'.format(floor=self.floor_num)), index=True,
            header=False, sep='\t')

    def figure_population_of_regions(self, shot_time):
        all_macs_traj = self.reload_all_combine_traj()
        combined_population_dict = self.generate_population_snapshot(traj_combine_df=all_macs_traj, shot_time=shot_time)
        # segments_flow_dict = self.generate_flow_snapshot(traj_combine=all_macs_traj,shot_time=shot_time,time_interval=time_interval,end_time=end_time)
        # print('combined_population_dict:\n',combined_population_dict)
        print('shot_time', shot_time)
        # print('segments_flow_dict:\n',segments_flow_dict)
        regions = self.itdv.regions
        doors = self.itdv.doors
        regions_pop = []
        region_ids = list(map(lambda x: x['id'], regions))
        # make up the missing zero population regions
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

    def figure_distance_adjacency_matrix(self, regions_ids):
        pass


shot_time = datetime.strptime('2017-01-05 11:00:00', '%Y-%m-%d %H:%M:%S')
end_time = datetime.strptime('2017-01-05 13:00:00', '%Y-%m-%d %H:%M:%S')
time_interval = 10

tjr = TrajectoryRecover()
recovered_location_20170105_2nd_floor = tjr.recover_all_macs_traj()
tjr.generate_pop_data(shot_time, time_interval, end_time)

