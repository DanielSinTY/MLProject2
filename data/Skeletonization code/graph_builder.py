from collections import deque
from graph import Graph
from constant import SKELETONIZE
from candidate_point import CandidatePoints

class BifurcateStack:

    def __init__(self):
        self.top = 0
        self.parent_node = [-1] * SKELETONIZE.STORAGE_MAX
        self.item = [[] for _ in range(SKELETONIZE.STORAGE_MAX)]

    def push(self, in_stack_points, parent_index):
        if self.top >= SKELETONIZE.STORAGE_MAX:
            raise SimpleException("BifurcateStack: Stack Full Error!")
        self.parent_node[self.top] = parent_index
        self.item[self.top] = in_stack_points.copy()
        self.top += 1

class GraphBuilder:
    nearby_points = []

    def __init__(self):
        self.selected_points = []
        self.point_mark = []
        self.stack = BifurcateStack()
        self.graph = Graph()
        self.candidate_points = CandidatePoints()

    def search_unused_nearby_points(self, selected_points):
        GraphBuilder.nearby_points = []
        self.candidate_points.query_points_nearby(selected_points, GraphBuilder.nearby_points)
        selected_points.clear()
        for point in GraphBuilder.nearby_points:
            if self.point_mark[point] < 0:
                selected_points.append(point)

    def check_connectivity(self, selected_points, parent_node):
        temp = []
        if len(selected_points) == 2:
            if not self.candidate_points.is_point_nearby(selected_points[0], selected_points[1]):
                temp.clear()
                temp.append(selected_points[1])
                self.stack.push(temp, parent_node)
                selected_points.pop()
            return
        
        point_stack = []
        first_branch = []
        point_used = [False] * len(selected_points)
        
        for i in range(len(selected_points)):
            if not point_used[i]:
                point_stack.append(i)
                temp.clear()
            while point_stack:
                temp_index = point_stack.pop()
                point_used[temp_index] = True
                temp.append(selected_points[temp_index])
                for j in range(i + 1, len(selected_points)):
                    if not point_used[j] and self.candidate_points.is_point_nearby(selected_points[temp_index], selected_points[j]):
                        point_stack.append(j)
                        break
            if not first_branch:
                first_branch = temp
            else:
                self.stack.push(temp, parent_node)
        
        selected_points[:] = first_branch

    def search_further_point(self, points_in_current_node, current_node_index):
        direction_vec = [0, 0]
        current_node = self.graph.get_node(current_node_index)
        if current_node.degree != 1:
            return -1
        last_node = self.graph.get_node(current_node.adjacent[0])
        for i in range(2):
            direction_vec[i] = current_node.center[i] - last_node.center[i]
        direction_norm_square = direction_vec[0] ** 2 + direction_vec[1] ** 2
        if direction_norm_square == 0:
            return -1
        
        base_point = current_node.center.copy()
        for point in points_in_current_node:
            temp_center = self.candidate_points.get_point(point)
            projection_len = direction_vec[0] * (temp_center[0] - base_point[0]) + direction_vec[1] * (temp_center[1] - base_point[1])
            if projection_len > 0:
                base_point[0] += direction_vec[0] * projection_len / direction_norm_square
                base_point[1] += direction_vec[1] * projection_len / direction_norm_square
        
        base_point[0] += SKELETONIZE.ANGLE_ERROR * direction_vec[0]
        base_point[1] += SKELETONIZE.ANGLE_ERROR * direction_vec[1]
        return self.candidate_points.query_points_by_pointer(base_point, direction_vec)

    def search_next_points(self):
        current_node = self.graph.get_node_num() - 1
        points_in_current_node = self.selected_points.copy()

        self.search_unused_nearby_points(self.selected_points)
        if len(self.selected_points) > 1:
            self.check_connectivity(self.selected_points, current_node)
        
        
        if not self.selected_points:
            furthur_point = self.search_further_point(points_in_current_node, current_node)
            if furthur_point >= 0:
                if self.point_mark[furthur_point] >= 0:
                    self.graph.connect_node(self.point_mark[furthur_point], current_node)
                    
                else:
                    self.selected_points.append(furthur_point)
        
        if not self.selected_points:
            while self.stack.top > 0:
                self.stack.top -= 1
                if self.point_mark[self.stack.item[self.stack.top][0]] < 0:
                    self.selected_points = self.stack.item[self.stack.top]
                    current_node = self.stack.parent_node[self.stack.top]
                    break
        
        if not self.selected_points:
            for i in range(self.candidate_points.get_point_num()):
                if self.point_mark[i] < 0:
                    self.selected_points.append(i)
                    break
            current_node = -1
        
        if len(self.selected_points) == 1:
            self.search_unused_nearby_points(self.selected_points)
        
        if self.selected_points:
            for point in self.selected_points:
                self.point_mark[point] = self.graph.get_node_num()
            self.graph.add_node(self.candidate_points.get_center(self.selected_points), current_node)
            

    def connecting_end(self):
        for i in range(self.graph.get_node_num()):
            end_node_index = i
            while self.graph.get_node(end_node_index).degree == 1:
                end_node_points = [j for j in range(self.candidate_points.get_point_num()) if self.point_mark[j] == end_node_index]
                point_index = self.search_further_point(end_node_points, end_node_index)
                if point_index < 0 or self.point_mark[point_index] < 0:
                    break
                self.graph.connect_node(end_node_index, self.point_mark[point_index])
                end_node_index = self.point_mark[point_index]

    def convert_to_graph(self, candidate_points, skeleton_graph, pic_num_str):
        self.candidate_points = candidate_points
        self.graph = skeleton_graph
        self.graph.reset()
        point_num = self.candidate_points.get_point_num()
        self.point_mark = [-1] * point_num
        self.stack.top = 0
        self.selected_points = [0]

        while self.selected_points:
            self.search_next_points()
            
        
        self.connecting_end()
        del self.point_mark
