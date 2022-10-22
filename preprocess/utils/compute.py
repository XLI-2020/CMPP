#coding:utf-8
from db import RawDataReader
from topology_graph import IndoorTopoDoorVertex
from datetime import datetime,timedelta
from copy import copy
from collections import OrderedDict
import os
import json
import random

class TrajectoryToDoorFLow(object):

    def __init__(self):
        self.rdr = RawDataReader()
        self.itdv = IndoorTopoDoorVertex(floor_num=1)
        self.start_day = datetime.strptime('2017-01-05', '%Y-%m-%d')
        self.end_day = datetime.strptime('2017-01-06', '%Y-%m-%d')
        self.map_path = '../map_data/'
        self.building_map_dict = self.get_building_map_dict()
        self.door_to_regions_relation_path = '../relation_data/'
        self.door_to_regions_relation_dict = self.get_door_to_regions_relation()

        pass

    def load_map_data(self,file_name):
        path = os.path.join(self.map_path, file_name)
        with open(path, 'r+') as file:
            map_data = json.loads(file.read())
        return map_data

    def get_building_map_dict(self):
        building_map_dict = {}
        for floor in range(1,8):
            filename = str(floor) + '.json'
            floor_map = self.load_map_data(filename)
            building_map_dict[floor] = floor_map
        return building_map_dict

    def get_door_to_regions_relation(self):
        door_to_regions_dict = {}
        for floor in range(1,8):
            filename = 'door_to_regions_%s.json'%(str(floor))
            path = os.path.join(self.door_to_regions_relation_path,filename)
            with open(path,'r+') as file:
                door_to_region_relation = json.loads(file.read())
            door_to_regions_dict[floor] = door_to_region_relation
        return door_to_regions_dict
    def load_raw_trajectory_data(self,day=None,mac=None):
        days_trajectory_dict = self.rdr.read_raw_data(self.start_day, self.end_day)
        if day is not None:
            day_trajectory_dict = days_trajectory_dict.get(day,None)
            if mac is not None:
                mac_trajectory_dict = day_trajectory_dict.get(mac,None)
                return mac_trajectory_dict
            else:
                return day_trajectory_dict
        else:
            if mac is None:
                return days_trajectory_dict
            else:
                one_mac_trajectory_data = list(list(days_trajectory_dict.values())[0].values())[1]
                # mac_trajectory_set_ordered = sorted(one_mac_trajectory_data, key=lambda x: x[0], reverse=False)
                return one_mac_trajectory_data

    def process_one_mac_trajectory_data(self):
        one_mac_trajectory_dict = self.load_raw_trajectory_data(mac=1)
        print('samples',one_mac_trajectory_dict)
        start_time = datetime.strptime('2017-01-05','%Y-%m-%d') + timedelta(hours=8)
        end_time = datetime.strptime('2017-01-05','%Y-%m-%d') + timedelta(hours=23)
        time_interval = timedelta(minutes=10)
        traj_segments = self.segment_traj_data_by_time_interval(traj_data=one_mac_trajectory_dict,start_time=start_time,end_time=end_time,time_interval=time_interval)
        print('2',traj_segments)
        traj_segments_items = list(traj_segments.items())
        segments_door_flow_dict = OrderedDict()
        timestamp_door_flow_dict = OrderedDict()
        for segment_index in range(len(traj_segments_items)):
            segment = traj_segments_items[segment_index]
            segment_ti = segment[0]
            segment_traj = segment[1]
            segment_traj_items = list(segment_traj.items())
            segment_door_flow_dict = OrderedDict()
            if segment_traj_items:
                for timestamp_index in range(len(segment_traj_items)):
                    timestamp_location = segment_traj_items[timestamp_index]
                    timestamp = timestamp_location[0]
                    location = timestamp_location[1]
                    # timestamp_door_flow_dict = {}
                    if timestamp_index == 0:
                        if segment_index == 0 or not traj_segments_items[segment_index-1][1]:
                            print('segment_index:%d segment_index == 0 or not traj_segments_items[segment_index-samples][samples]'%segment_index,traj_segments_items[segment_index-1][1])
                            continue
                        else:
                            pre_segment = traj_segments_items[segment_index-1]
                            pre_segment_ti = pre_segment[0]
                            pre_segment_traj = pre_segment[1]
                            pre_timestamp_location = list(pre_segment_traj.items())[-1]
                            pre_timestamp = pre_timestamp_location[0]
                            pre_location = pre_timestamp_location[1]
                            door_flow_dict = self.compute_door_flow_in_adjacent_locaions(pre_timestamp,pre_location,timestamp,location)
                            print('segment_index:%d,segment_ti,timestamp_index,pre_timestamp,timestamp,door_flow_dict\n'%(segment_index),segment_ti,timestamp_index,pre_timestamp,timestamp,door_flow_dict)
                            timestamp_door_flow_dict[(pre_timestamp,timestamp)] = door_flow_dict
                            self.update_segment_door_flow(door_flow_dict,segment_door_flow_dict)
                    else:
                        pre_timestamp_location = segment_traj_items[timestamp_index-1]
                        pre_timestamp = pre_timestamp_location[0]
                        pre_location = pre_timestamp_location[1]
                        door_flow_dict = self.compute_door_flow_in_adjacent_locaions(pre_timestamp, pre_location,timestamp, location)
                        print('segment_index:%d,segment_ti,timestamp_index,pre_timestamp,timestamp,door_flow_dict\n'%(segment_index),segment_ti, timestamp_index, pre_timestamp, timestamp, door_flow_dict)
                        timestamp_door_flow_dict[(pre_timestamp, timestamp)] = door_flow_dict
                        self.update_segment_door_flow(door_flow_dict, segment_door_flow_dict)
                segments_door_flow_dict[segment_ti] = segment_door_flow_dict
            else:
                print('segment_index:%d no segment_traj_items'%(segment_index))
                # segments_door_flow_dict[segment_ti] = OrderedDict()
                continue
        return segments_door_flow_dict,timestamp_door_flow_dict

    def update_segment_door_flow(self,door_flow_dict,segment_door_flow_dict):
        if door_flow_dict:
            for door_id,flow_value in door_flow_dict.items():
                if door_id in segment_door_flow_dict:
                    segment_door_flow_dict[door_id] = segment_door_flow_dict[door_id] + flow_value
                else:
                    segment_door_flow_dict[door_id] = flow_value

    def segment_traj_data_by_time_interval(self,traj_data=None,start_time=None,end_time=None,time_interval=None):
        one_mac_trajectory_dict = traj_data
        start_time = start_time
        end_time = end_time
        time_interval = time_interval
        curr_time = copy(start_time)
        segments_traj = OrderedDict()
        # print('3',one_mac_trajectory_dict)
        while curr_time <= (end_time-time_interval):
            segment_ti = (curr_time,curr_time+time_interval)
            segment_traj = OrderedDict(list(filter(lambda x:x[0] >= curr_time and x[0] < (curr_time+time_interval),list(one_mac_trajectory_dict.items()))))
            segments_traj[segment_ti] = segment_traj
            curr_time = curr_time + time_interval
        return segments_traj

    def compute_door_flow_in_adjacent_locaions(self,pre_timestamp=None,pre_location=None,timestamp=None,location=None):
        pre_floor_num = pre_location[2]
        curr_floor_num = location[2]
        if pre_floor_num != 1 or curr_floor_num != 1:
            return {}
        pre_location_in_region_id = self.find_locaton_in_which_region(pre_location)
        curr_location_in_region_id = self.find_locaton_in_which_region(location)
        if not pre_location_in_region_id or not curr_location_in_region_id:
            return {}
        else:
            if len(pre_location_in_region_id) > 1:
                print('pre_location_in_region_id len more than samples>compute_door_flow_in_adjacent_locaions')
            if len(curr_location_in_region_id) > 1:
                print('location_in_region_id len more than samples>compute_door_flow_in_adjacent_locaions')
            pre_region_id = pre_location_in_region_id[0]
            current_region_id = curr_location_in_region_id[0]
            if pre_region_id == current_region_id:
                return {}
            region_pair = '-'.join([str(pre_region_id),str(current_region_id)])
            door_to_regions_relation = self.door_to_regions_relation_dict.get(pre_floor_num)#pre_floor_num=curr_floor_num
            if region_pair in door_to_regions_relation:
                door_id_list = door_to_regions_relation[region_pair]
                if len(door_id_list) > 1:
                    door_id = random.choice(door_id_list)
                else:
                    door_id = door_id_list[0]
                return {door_id:1}
            else:
                simple_paths = self.itdv.find_simple_path_p2p(pre_location,location)
                indoor_topo = self.itdv.construct_indoor_topo_p2p(pre_location,location)
                door_flow_dict = self.itdv.figure_door_flow_p2p(G=indoor_topo,paths=simple_paths)
                return door_flow_dict


    def find_locaton_in_which_region(self,location):
        x,y,floor_num = location
        floor_map_dict = self.building_map_dict.get(floor_num)
        regions = list(map(lambda x:[x.get('id'),x.get('points')],floor_map_dict["regions"]))
        regions_of_location = list(filter(lambda z:self.location_in_region(z[1],x,y),regions))
        if regions_of_location:
            regions_id = list(map(lambda x:x[0],regions_of_location))
        else:
            regions_id=[]
        return regions_id


    def location_in_region(self,region_points,x,y):
        region_x_list = list(map(lambda x:x['x'],region_points))
        region_y_list = list(map(lambda x:x['y'],region_points))
        min_x, max_x = min(region_x_list),max(region_x_list)
        min_y, max_y = min(region_y_list),max(region_y_list)
        if x>=min_x and x<= max_x and y>=min_y and y<=max_y:
            return True
        else:
            return False

    def recover_one_mac_traj(self):
        one_mac_trajectory_dict = self.load_raw_trajectory_data(mac=1)
        print('samples', one_mac_trajectory_dict)
        one_mac_trajectory_dict_items = one_mac_trajectory_dict.items()
        for timestamp_index in range(len(one_mac_trajectory_dict_items)-1):
            pass







