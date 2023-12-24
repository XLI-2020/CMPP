#coding:utf-8
import networkx as nx
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import os
from itertools import combinations
from data_util import DataUtil
from itertools import chain
import copy
import random
from collections import OrderedDict


class IndoorTopoDoorVertex(object):
    def __init__(self,floor_num):
        self.map_path = '../map_data/'
        self.file = str(floor_num) + '.json'
        self.file_path = os.path.join(self.map_path,self.file)
        self.map_dict = self.load_map_data(self.file_path)
        self.regions = self.map_dict['regions']
        doors = self.map_dict['doors']
        virtual_doors = self.map_dict['virtualDoors']
        self.doors = doors + virtual_doors
        self.G = self.construct_weight_edges()
        self.s_max = 1.53

    def load_map_data(self,filepath):
        with open(filepath,'r+') as file:
            map_dict = json.load(file)
            return map_dict

    def construct_weight_edges(self):
        G = nx.Graph()
        regions = list(map(lambda x:[x['id'],x.get('connectedDoorsID',[])],self.regions))
        total_weighted_edges = []
        for region in regions:
            region_id = region[0]
            connectedDoorsID = region[1]
            if not connectedDoorsID:
                continue
            edges = combinations(connectedDoorsID,2)
            edges = list(map(lambda x:sorted(x,reverse=False),edges))
            weighted_edges = list(map(lambda x:self.trans_to_weight_edges(x),edges))
            weighted_edges = list(filter(lambda x:x,weighted_edges))
            total_weighted_edges.extend(weighted_edges)
        G.add_weighted_edges_from(total_weighted_edges)
        return G

    def find_simple_path_d2d(self,start_door,end_door):
        indoor_topo = self.construct_weight_edges()
        shorest_path_len = nx.shortest_path_length(G=indoor_topo,source=start_door,target=end_door)
        # print('shorest_path_len',shorest_path_len)
        simple_paths = nx.all_simple_paths(G=indoor_topo,source=start_door,target=end_door,cutoff=2*shorest_path_len)
        simple_paths_list = copy.copy(list(simple_paths))
        # print('samples',list(simple_paths_list))
        # print('len of simple_paths',len(list(simple_paths_list)))
        normal_simple_path = list(filter(lambda x:self.is_normal_path(list(x)),list(simple_paths_list)))
        return normal_simple_path

    def construct_indoor_topo_p2p(self,start_point,end_point):
        start_region_id = self.find_locaton_in_which_region(start_point)
        end_region_id = self.find_locaton_in_which_region(end_point)
        indoor_topo = self.construct_weight_edges()
        if len(start_region_id) > 1:
            print('start_region_id len more than samples>construct_indoor_topo_p2p')
        if len(end_region_id) > 1:
            print('end_region_id len more than samples>construct_indoor_topo_p2p')
        start_region_id = start_region_id[0]
        end_region_id = end_region_id[0]
        start_region_info = list(filter(lambda x:x.get('id') == start_region_id,self.regions))
        start_region_connected_doors = start_region_info[0]['connectedDoorsID']
        s_total_edges = []
        for s_door_id in start_region_connected_doors:
            s_door_point = self.find_door_point(s_door_id)
            s_edge_weight = DataUtil.euclidean_dist_points(s_door_point,start_point[:2])
            s_weighted_edge = ('s',s_door_id,s_edge_weight)
            s_total_edges.append(s_weighted_edge)
        indoor_topo.add_weighted_edges_from(s_total_edges)

        end_region_info = list(filter(lambda x:x.get('id') == end_region_id,self.regions))
        end_region_connected_doors = end_region_info[0]['connectedDoorsID']
        e_total_edges =[]
        for e_door_id in end_region_connected_doors:
            e_door_point = self.find_door_point(e_door_id)
            e_edge_weight = DataUtil.euclidean_dist_points(e_door_point,end_point[:2])
            e_weighted_edge = ('e',e_door_id,e_edge_weight)
            e_total_edges.append(e_weighted_edge)
        indoor_topo.add_weighted_edges_from(e_total_edges)
        return indoor_topo

    def find_simple_path_p2p(self, start_point,start_timestamp, end_point, end_timestamp):
        S_max = 1.53
        time_interval = (end_timestamp - start_timestamp).total_seconds()
        indoor_topo = self.construct_indoor_topo_p2p(start_point,end_point)
        start_region_id = self.find_locaton_in_which_region(start_point)
        end_region_id = self.find_locaton_in_which_region(end_point)
        # print('region id',start_region_id,end_region_id)
        if len(start_region_id) > 1:
            print('start_region_id len more than samples>find_simple_path_p2p')
        if len(end_region_id) > 1:
            print('end_region_id len more than samples>find_simple_path_p2p')
        start_region_id = start_region_id[0]
        end_region_id = end_region_id[0]
        try:
            shorest_path_len = nx.shortest_path_length(G=indoor_topo, source='s', target='e')
        except:
            shorest_path_len = 1
        if shorest_path_len <= 3:
            path_length_limit = 2*shorest_path_len
        elif shorest_path_len >3 and shorest_path_len <= 6:
            path_length_limit = 6
        else:
            path_length_limit = copy.copy(shorest_path_len)

        simple_paths = nx.all_simple_paths(G=indoor_topo, source='s', target='e',
                                           cutoff=path_length_limit)
        try:
            dijstra_path_length = nx.dijkstra_path_length(G=indoor_topo,source='s', target='e')
        except:
            dijstra_path_length = 5
        # print('dijstra_path_length',dijstra_path_length)
        simple_paths_list = copy.copy(list(simple_paths))
        # print('samples',list(simple_paths_list))
        # print('len of simple_paths',len(list(simple_paths_list)))
        normal_simple_path = list(filter(lambda x: self.is_normal_path_p2p(list(x),start_region_id,end_region_id), list(simple_paths_list)))
        normal_distance_path = list(filter(lambda x:self.figure_path_length_p2p(G=indoor_topo,path=x) <= (time_interval*S_max),normal_simple_path))

        return normal_distance_path

    def figure_nonadjacent_points_paths(self,start_point,start_timestamp,end_point,end_timestamp):
        simple_paths = self.find_simple_path_p2p(start_point,start_timestamp, end_point, end_timestamp)
        indoor_topo = self.construct_indoor_topo_p2p(start_point,end_point)
        simple_paths_weights_dict = self.figure_simple_path_weight(G=indoor_topo,paths=simple_paths)
        all_paths_dict = {}
        # print('len of simple path',len(simple_paths))
        for path in simple_paths:
            path_length = self.figure_path_length_p2p(G=indoor_topo,path=path)
            time_delta = (end_timestamp - start_timestamp).total_seconds()
            path_directions = self.find_path_directions(path=path,start_point=start_point,end_point=end_point)
            path_info_list = []
            cur_node_info_dict = {}
            pre_node_info_dict = {}
            for node in path:
                if not pre_node_info_dict:
                    pre_node_info_dict['node_id'] = node
                    pre_node_info_dict['timestamp'] = start_timestamp
                    # node_info_dict['point'] = start_point
                    path_info_list.append(pre_node_info_dict)
                else:
                    if node != 'e':
                        # print('test',node,pre_node_info_dict)
                        cur_node_info_dict['node_id'] = node
                        slice_length = indoor_topo[pre_node_info_dict['node_id']][node]['weight']
                        distant_slice_length = indoor_topo[node]['e']['weight']

                        lower_bound = slice_length/self.s_max + pre_node_info_dict['timestamp']
                        upper_bound = end_timestamp - distant_slice_length/self.s_max
                        current_node_timestamp = random.sample(range(int(lower_bound), int(upper_bound)), 1)
                        # slice_time = (slice_length * time_delta) / float(path_length)
                        # cur_node_info_dict['timestamp'] = pre_node_info_dict['timestamp'] + timedelta(
                        #     seconds=slice_time)
                        cur_node_info_dict['timestamp'] = current_node_timestamp
                        path_info_list.append(cur_node_info_dict)
                        pre_node_info_dict = copy.copy(cur_node_info_dict)
                        cur_node_info_dict = {}
                    else:
                        cur_node_info_dict['node_id'] = node
                        cur_node_info_dict['timestamp'] = end_timestamp
                        path_info_list.append(cur_node_info_dict)
                        pre_node_info_dict = {}
                        cur_node_info_dict = {}
                    # if node != 'e':
                    #     # print('test',node,pre_node_info_dict)
                    #     cur_node_info_dict['node_id'] = node
                    #     slice_length = indoor_topo[pre_node_info_dict['node_id']][node]['weight']
                    #     slice_time = (slice_length*time_delta)/float(path_length)
                    #     cur_node_info_dict['timestamp'] = pre_node_info_dict['timestamp'] + timedelta(seconds=slice_time)
                    #     path_info_list.append(cur_node_info_dict)
                    #     pre_node_info_dict = copy.copy(cur_node_info_dict)
                    #     cur_node_info_dict = {}
                    # else:
                    #     cur_node_info_dict['node_id'] = node
                    #     cur_node_info_dict['timestamp'] = end_timestamp
                    #     path_info_list.append(cur_node_info_dict)
                    #     pre_node_info_dict = {}
                    #     cur_node_info_dict = {}

            assert len(path_info_list)==len(path_directions),'path_info_list unequal path_directions'
            for node_index in range(len(path_info_list)):
                path_info_list[node_index]['direction'] = path_directions[node_index]
            path_info_dict = {'path_weight':simple_paths_weights_dict[tuple(path)],'path_info_list':path_info_list}
            all_paths_dict[tuple(path)] = path_info_dict
        return all_paths_dict

    def find_random_path(self,simple_paths_weights_dict, rn, ordered_prob_list):
        for i, item in enumerate(simple_paths_weights_dict.items()):
            if rn <= sum(ordered_prob_list[:i + 1]) * 100:
                return [list(item[0]),]

    def figure_nonadjacent_points_paths_random(self,start_point,start_timestamp,end_point,end_timestamp):
        simple_paths = self.find_simple_path_p2p(start_point,end_point)
        indoor_topo = self.construct_indoor_topo_p2p(start_point,end_point)
        simple_paths_weights_dict = self.figure_simple_path_weight(G=indoor_topo,paths=simple_paths)
        simple_paths_weights_dict = OrderedDict(simple_paths_weights_dict)
        rn = random.choice(range(100))
        ordered_prob_list = list(map(lambda x:x[1], list(simple_paths_weights_dict.items())))
        # random select path
        print('simple_paths_weights_dict',simple_paths_weights_dict)
        print('ordered_prob_list',ordered_prob_list)
        print('rn',rn)
        random_path = self.find_random_path(simple_paths_weights_dict,rn,ordered_prob_list)
        print('random_path',random_path)
        all_paths_dict = {}
        for path in random_path:
            path_length = self.figure_path_length_p2p(G=indoor_topo,path=path)
            time_delta = (end_timestamp - start_timestamp).total_seconds()
            path_directions = self.find_path_directions(path=path,start_point=start_point,end_point=end_point)
            # print('path start_point end_point', path_directions,start_point,end_point)
            path_info_list = []
            cur_node_info_dict = {}
            pre_node_info_dict = {}
            for node in path:
                if not pre_node_info_dict:
                    pre_node_info_dict['node_id'] = node
                    pre_node_info_dict['timestamp'] = start_timestamp
                    # node_info_dict['point'] = start_point
                    path_info_list.append(pre_node_info_dict)
                else:
                    if node != 'e':
                        # print('test',node,pre_node_info_dict)
                        cur_node_info_dict['node_id'] = node
                        slice_length = indoor_topo[pre_node_info_dict['node_id']][node]['weight']
                        slice_time = (slice_length*time_delta)/float(path_length)
                        cur_node_info_dict['timestamp'] = pre_node_info_dict['timestamp'] + timedelta(seconds=slice_time)
                        path_info_list.append(cur_node_info_dict)
                        pre_node_info_dict = copy.copy(cur_node_info_dict)
                        cur_node_info_dict = {}
                    else:
                        cur_node_info_dict['node_id'] = node
                        cur_node_info_dict['timestamp'] = end_timestamp
                        path_info_list.append(cur_node_info_dict)
                        pre_node_info_dict = {}
                        cur_node_info_dict = {}

            assert len(path_info_list)==len(path_directions),'path_info_list unequal path_directions'
            for node_index in range(len(path_info_list)):
                path_info_list[node_index]['direction'] = path_directions[node_index]
            path_info_dict = {'path_weight':1,'path_info_list':path_info_list}
            all_paths_dict[tuple(path)] = path_info_dict
        return all_paths_dict


    def figure_simple_path_weight(self,G,paths):
        path_weight_dict = {tuple(path): 1 / float(self.figure_path_length_p2p(G, path)) for path in paths}
        path_norm_weight_dict = {k: round(v / sum(path_weight_dict.values()), 8) for k, v in path_weight_dict.items()}
        return path_norm_weight_dict


    def find_path_directions(self,path,start_point,end_point):
        start_region = self.find_locaton_in_which_region(start_point)[0]
        end_region = self.find_locaton_in_which_region(end_point)[0]
        assert len(path) >=3, 'path length less than 3 nodes'
        if len(path) == 3:
            path_directions = [None,[start_region,end_region],None]
        else:
            door_path = path[1:-1]
            door_path_regions = [start_region]
            for i in range(len(door_path)-1):
                regions_of_curdoor = self.find_door_connected_regions(door_path[i])
                regions_of_nextdoor = self.find_door_connected_regions(door_path[i+1])
                # print('door',door_path[i],door_path[i+1])
                # print('region',regions_of_nextdoor,regions_of_curdoor)
                common_regions = list(set(regions_of_curdoor).intersection(set(regions_of_nextdoor)))
                assert common_regions, 'adjacent doors have no common_regions!'
                if len(common_regions) > 1:
                    print('common_regions more than samples>find_path_directions',len(common_regions))
                common_region = random.choice(common_regions)
                door_path_regions.append(common_region)
            door_path_regions.append(end_region)
            door_path_directions =[[door_path_regions[index],door_path_regions[index+1]] for index in range(len(door_path_regions)-1)]
            path_directions = copy.copy(door_path_directions)
            path_directions.insert(0,None)
            path_directions.append(None)
            assert len(path_directions) == len(path)
        return path_directions

    def figure_adjacent_points_path(self,start_point,start_timestamp,end_point,end_timestamp):
        # print('adjc')
        # simple_paths = self.find_simple_path_p2p(start_point, end_point)
        indoor_topo = self.construct_indoor_topo_p2p(start_point, end_point)
        # print('nodes',indoor_topo.nodes)
        # print('edges',indoor_topo.edges)

        start_region = self.find_locaton_in_which_region(start_point)[0]
        # print('start_region',start_region)
        end_region = self.find_locaton_in_which_region(end_point)[0]
        # print('end_region',end_region)
        shortest_path = nx.shortest_path(G=indoor_topo,source='s',target='e')
        # print('shortest_path',shortest_path)
        assert len(shortest_path)==3, 'shortest_path unequal 3 for adjacent points'
        path_length = self.figure_path_length_p2p(G=indoor_topo,path=shortest_path)
        time_delta = (end_timestamp - start_timestamp).total_seconds()
        node_id = shortest_path[1]
        slice_length = indoor_topo['s'][node_id]['weight']
        slice_time = (slice_length*time_delta)/float(path_length)
        path_info_dict = {}
        path_info_dict['node_id'] = node_id
        path_info_dict['timestamp'] = start_timestamp + timedelta(seconds=slice_time)
        path_info_dict['direction'] = [start_region,end_region]
        # print('path_info_dict',path_info_dict)
        return path_info_dict

    def figure_adjacent_points_path_random(self,start_point,start_timestamp,end_point,end_timestamp):
        # simple_paths = self.find_simple_path_p2p(start_point, end_point)
        indoor_topo = self.construct_indoor_topo_p2p(start_point, end_point)
        start_region = self.find_locaton_in_which_region(start_point)[0]
        # print('start_region',start_region)
        end_region = self.find_locaton_in_which_region(end_point)[0]
        # print('end_region',end_region)
        shortest_path = nx.shortest_path(G=indoor_topo,source='s',target='e')
        # print('shortest_path',shortest_path)
        assert len(shortest_path)==3, 'shortest_path unequal 3 for adjacent points'
        path_length = self.figure_path_length_p2p(G=indoor_topo,path=shortest_path)
        time_delta = (end_timestamp - start_timestamp).total_seconds()
        node_id = shortest_path[1]
        slice_length = indoor_topo['s'][node_id]['weight']
        slice_time = (slice_length*time_delta)/float(path_length)
        #random time

        path_info_dict = {}
        path_info_dict['node_id'] = node_id
        path_info_dict['timestamp'] = start_timestamp + timedelta(seconds=slice_time)
        path_info_dict['direction'] = [start_region,end_region]
        # print('path_info_dict',path_info_dict)
        return path_info_dict

    def find_shortest_path_p2p(self,start_point, end_point):
        indoor_topo = self.construct_indoor_topo_p2p(start_point, end_point)
        start_region_id = self.find_locaton_in_which_region(start_point)
        end_region_id = self.find_locaton_in_which_region(end_point)
        if len(start_region_id) > 1:
            print('start_region_id len more than samples>find_shortest_path_p2p')
        if len(end_region_id) > 1:
            print('end_region_id len more than samples>find_shortest_path_p2p')
        start_region_id = start_region_id[0]
        end_region_id = end_region_id[0]
        shorest_path = nx.shortest_path(G=indoor_topo, source='s', target='e')
        return shorest_path

    def figure_path_length_p2p(self,G,path):
        path_length = sum([G[path[i]][path[i+1]]['weight'] for i in range(len(path)-1)])
        return path_length

    def figure_door_flow_p2p(self,G,paths):
        door_flow_dict = {}
        path_weight_dict = {tuple(path):1/float(self.figure_path_length_p2p(G,path)) for path in paths}
        path_norm_weight_dict = {k: round(v/sum(path_weight_dict.values()),4) for k,v in path_weight_dict.items()}
        # print('path_norm_weight_dict',path_norm_weight_dict)
        for k,v in path_norm_weight_dict.items():
            for node in k:
                if node !='s' and node !='e':
                    if node in door_flow_dict:
                        door_flow_dict[node] = door_flow_dict[node] + v
                    else:
                        door_flow_dict[node] = v
        return door_flow_dict


    def is_normal_path_p2p(self,path,s_region_id,e_region_id):
        normal_flag = True
        for node_index in range(0, len(path) - 2):
            if path[node_index] == 's':
                region0 = [s_region_id]
            else:
                region0 = self.find_door_connected_regions(path[node_index])
            if path[node_index+2] == 'e':
                region2 = [e_region_id]
            else:
                region2 = self.find_door_connected_regions(path[node_index+2])
            region1 = self.find_door_connected_regions(path[node_index+1])
            if set(region0).intersection(set(region1)).intersection(set(region2)):
                normal_flag = False
                break
        return normal_flag

    def is_normal_path(self,path):
        normal_flag = True
        for node_index in range(0,len(path)-2):
             if self.if_doors_have_common_region(path[node_index],path[node_index+1],path[node_index+2]):
                 normal_flag = False
                 break
        return normal_flag

    def if_doors_have_common_region(self,door_id0,door_id1,door_id2):
        door0_connectedRegionsID = self.find_door_connected_regions(door_id0)
        door1_connectedRegionsID = self.find_door_connected_regions(door_id1)
        door2_connectedRegionsID = self.find_door_connected_regions(door_id2)
        if set(door0_connectedRegionsID).intersection(set(door1_connectedRegionsID)).intersection(set(door2_connectedRegionsID)):
            return True
        else:
            return False

    def trans_to_weight_edges(self,edge):
        if edge:
            start_door = edge[0]
            end_door = edge[1]
            start_point = self.find_door_point(start_door)
            end_point = self.find_door_point(end_door)
            if start_point and end_point:
                eu_distance = DataUtil.euclidean_dist_points(start_point,end_point)
                return (start_door,end_door,eu_distance)
            else:
                return ()
        else:
            return ()

    def find_door_point(self,door_id):
        door_info_list = list(filter(lambda x:x.get('id',None) == door_id, self.doors))
        if door_info_list:
            if len(door_info_list) > 1:
                print('find_door_point door_info_list more than samples')
            door_info = door_info_list[0]
            line = door_info.get('line')
            door_point = [(line['x1']+line['x2'])/2, (line['y1']+line['y2'])/2]
            return door_point
        else:
            return []
    def find_door_connected_regions(self,door_id):
        door_info_list = list(filter(lambda x:x.get('id',None) == door_id, self.doors))
        if door_info_list:
            if len(door_info_list) > 1:
                print('find_door_connected_regions door_info_list more than samples')
            door_info = door_info_list[0]
            connectedRegionsID = door_info.get('connectedRegionsID')
            return connectedRegionsID
        else:
            return []

    def find_locaton_in_which_region(self,location):
        x,y,floor_num = location
        floor_map_dict = copy.copy(self.map_dict)
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

def validate():
    itdv = IndoorTopoDoorVertex(floor_num=1)
    # print(itdv.map_dict)

    print('samples',len(itdv.doors))
    indoor_topo = itdv.construct_weight_edges()
    print('2',len(indoor_topo.nodes))

    door_pair_list = list(map(lambda x:list(combinations(x['connectedDoorsID'],2)),itdv.regions))
    door_pair_list = list(chain(*door_pair_list))
    door_pair_set = list(set(list(map(lambda x:tuple(sorted(x)),door_pair_list))))
    print('3',len(door_pair_set))

    print(indoor_topo.edges)
    print('4',len(indoor_topo.edges))






