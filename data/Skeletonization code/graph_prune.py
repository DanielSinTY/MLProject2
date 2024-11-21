from enum import Enum, auto
import numpy as np
import math

from graph import Graph
from graph_structure import GraphStructure, GraphStructureNode
from root_smooth import RootSmooth
from constant import WORM, SKELETONIZE
from utils import WORM_SHAPE, worm_shape, SelectMinimum, calculate_curve_clockwise, calc_clockwise_angle, point_dist_square
from backbone import Backbone

NUM = 11
LARGE_SCALE_PARTITION_NUM = 10

class BACKBONE_STATE(Enum):
    REVERSE = 1
    ABANDON = 2
    FORWARD = 3

class GraphPrune:
    def __init__(self):
        self.graph = Graph()
        self.graph_structure = None
        self.node_available = None
        self.node_num = 0
        self.structure_node_num = 0
        self.structure_node_list = [GraphStructureNode() for _ in range(self.node_num)]
        self.root_smooth = RootSmooth()
        self.wormShape = "Normal"

    def calculate_turning_angle(self, current_cood, last_cood):
        angle = abs(math.atan2(current_cood[0], current_cood[1]) - math.atan2(last_cood[0], last_cood[1]))
        if angle > WORM.PI:
            angle = 2 * WORM.PI - angle
        return angle

    def same_direction(self, cline, last_backbone):
        p0 = self.graph.get_node(cline[0]).center
        p2 = self.graph.get_node(cline[-1]).center
        last_start = last_backbone.cood[0]
        last_end = last_backbone.cood[-1]
        return (p0[0] - p2[0]) * (last_start[0] - last_end[0]) + (p0[1] - p2[1]) * (last_start[1] - last_end[1]) > 0

    def same_clockwise(self, cline, last_backbone, start_2=None, end_2=None):
        if start_2 is None or end_2 is None:
            cline_data = np.zeros((NUM, 2))
            last_data = np.zeros((NUM, 2))
            for i in range(NUM - 1):
                current_index = i * (len(cline) // (NUM - 1))
                last_index = i * (last_backbone.length // (NUM - 1))
                cline_data[i] = self.graph.get_node(cline[current_index]).center
                last_data[i] = last_backbone.cood[last_index]
            cline_data[NUM - 1] = self.graph.get_node(cline[-1]).center
            last_data[NUM - 1] = last_backbone.cood[last_backbone.length - 1]
            current_clockwise = calculate_curve_clockwise(cline_data, NUM)
            last_clockwise = calculate_curve_clockwise(last_data, NUM)
            return current_clockwise == last_clockwise
        else:
            p0 = self.graph.get_node(cline[0]).center
            p1 = self.graph.get_node(cline[len(cline) // 3]).center
            p2 = self.graph.get_node(cline[len(cline) * 2 // 3]).center
            l0 = last_backbone.cood[start_2]
            l1 = last_backbone.cood[(start_2 * 2 + end_2) // 3]
            l2 = last_backbone.cood[(start_2 + end_2 * 2) // 3]
            angle_p = calc_clockwise_angle(p0, p1, p2)
            angle_l = calc_clockwise_angle(l0, l1, l2)
            return (angle_p < WORM.PI) == (angle_l < WORM.PI)


    def get_largest_subgraph(self):
        subgraph_num = 0
        stack_top = 0
        subgraph_mark = [-1] * self.node_num
        node_stack = [0] * self.node_num

        for i in range(self.node_num):
            if subgraph_mark[i] < 0:
                subgraph_num += 1
                node_stack[stack_top] = i
                stack_top += 1

            while stack_top > 0:
                stack_top -= 1
                current_index = node_stack[stack_top]
                subgraph_mark[current_index] = subgraph_num - 1
                current_node = self.graph.get_node(current_index)
                for j in range(current_node.degree):
                    if subgraph_mark[current_node.adjacent[j]] < 0:
                        node_stack[stack_top] = current_node.adjacent[j]
                        stack_top += 1

        subgraph_count = [0] * subgraph_num
        for i in range(self.node_num):
            subgraph_count[subgraph_mark[i]] += 1

        find_largest_subgraph = SelectMinimum(-subgraph_count[0], 0)
        for i in range(1, subgraph_num):
            find_largest_subgraph.renew(-subgraph_count[i], i)
        
        largest_subgraph = find_largest_subgraph.get_min_index()
        for i in range(self.node_num):
            self.node_available[i] = (subgraph_mark[i] == largest_subgraph)

    def rotate_to_next(self, last_node, current_node):
        current_graph_node = self.graph.get_node(current_node)
        last_graph_node = self.graph.get_node(last_node)
        direction = math.atan2(last_graph_node.center[0] - current_graph_node.center[0],
                               last_graph_node.center[1] - current_graph_node.center[1]) + SKELETONIZE.ANGLE_ERROR
        adjacent_select = SelectMinimum(WORM.INF, -1)
        for i in range(current_graph_node.degree):
            node_to = current_graph_node.adjacent[i]
            temp_center = self.graph.get_node(node_to).center
            angle_temp = math.atan2(temp_center[0] - current_graph_node.center[0],
                                    temp_center[1] - current_graph_node.center[1]) - direction
            while angle_temp < 0:
                angle_temp += 2 * WORM.PI
            adjacent_select.renew(angle_temp, i)
        last_node, current_node = current_node, current_graph_node.adjacent[adjacent_select.get_min_index()]
        return last_node, current_node

    def start_node_locate(self, first_node, second_node):
        find_leftmost_point = SelectMinimum(WORM.INF, -1)
        second_select = SelectMinimum(WORM.INF, -1)
        for i in range(self.node_num):
            if self.node_available[i]:
                find_leftmost_point.renew(self.graph.get_node(i).center[0], i)
        first_node = find_leftmost_point.get_min_index()
        leftmost_node = self.graph.get_node(first_node)
        for i in range(leftmost_node.degree):
            node_to = leftmost_node.adjacent[i]
            if not self.node_available[node_to]:
                continue
            temp_metric = math.atan2(self.graph.get_node(node_to).center[0] - leftmost_node.center[0],
                                     self.graph.get_node(node_to).center[1] - leftmost_node.center[1]) + WORM.PI / 2
            if temp_metric < 0:
                temp_metric += 2 * WORM.PI
            second_select.renew(temp_metric, i)
        second_node = leftmost_node.adjacent[second_select.get_min_index()]
        edge_num = 1
        while True:
            success, first_node, second_node = self.graph.get_node(second_node).select_next(first_node, second_node)
            if not success:
                break
            if edge_num > self.node_num:
                raise SimpleException("Circle Error!")
            edge_num += 1
        return self.rotate_to_next(first_node, second_node)

    def graph_structure_analyze(self, first_node, second_node):
        last_node, current_node = first_node, second_node
        edge = []
        while True:
            edge.clear()
            edge.append(last_node)
            edge.append(current_node)
            while True:
                success, last_node, current_node = self.graph.get_node(current_node).select_next(last_node, current_node)
                if not success:
                    break
                edge.append(current_node)

            self.graph_structure.add_edge(edge)
            while True:
                last_node, current_node = self.rotate_to_next(last_node, current_node)
                if last_node == first_node and current_node == second_node:
                    return
                success, last_node, current_node = self.graph_structure.move_to_other_end(last_node, current_node)
                if not success:
                    break
                    
    def delete_short_route(self):
        for i in range(self.structure_node_num):
            structure_node = self.structure_node_list[i]
            for j in range(structure_node.degree - 1):
                if structure_node.edges[0][0] == structure_node.edges[j][-1] and len(structure_node.edges[j]) <= 4:
                    self.graph_structure.delete_edge(structure_node.edges[j])
            if structure_node.degree == 1 and len(structure_node.edges[0]) <= 2:
                self.graph_structure.delete_edge(structure_node.edges[0])
        self.graph_structure.check_structure()

    def delete_shorter_routes_with_same_end(self):
        changed = False
        for i in range(self.structure_node_num):
            structure_node = self.structure_node_list[i]
            for j in range(structure_node.degree - 1):
                if structure_node.edges[j][-1] == structure_node.edges[0][0]:
                    continue
                for k in range(j + 1, structure_node.degree):
                    if structure_node.edges[j][-1] == structure_node.edges[k][-1]:
                        changed = True
                        delete_index = k if len(structure_node.edges[j]) > len(structure_node.edges[k]) else j
                        self.graph_structure.delete_edge(structure_node.edges[delete_index])
        if changed:
            self.graph_structure.check_structure()
        return changed

    def delete_branch_and_loopback_except_for_two_longest(self):
        changed = False
        branch_index = [0] * (self.structure_node_num * SKELETONIZE.DEGREE_MAX)
        branch_len = [0] * (self.structure_node_num * SKELETONIZE.DEGREE_MAX)
        branch_num = 0

        for i in range(self.structure_node_num):
            structure_node = self.structure_node_list[i]
            if structure_node.degree == 1:
                branch_index[branch_num] = i * SKELETONIZE.DEGREE_MAX
                branch_len[branch_num] = len(structure_node.edges[0])
                branch_num += 1
            else:
                for j in range(structure_node.degree):
                    if (structure_node.edges[0][0] == structure_node.edges[j][-1] and
                        structure_node.edges[j][1] < structure_node.edges[j][-2]):
                        branch_index[branch_num] = i * SKELETONIZE.DEGREE_MAX + j
                        branch_len[branch_num] = len(structure_node.edges[j]) // 2
                        branch_num += 1

        if branch_num > 2:
            changed = True
            longest_branch = SelectMinimum(WORM.INF, -1)
            second_longest_branch = SelectMinimum(WORM.INF, -1)
            for i in range(branch_num):
                longest_branch.renew(-branch_len[i], i)
            for i in range(branch_num):
                if i != longest_branch.get_min_index():
                    second_longest_branch.renew(-branch_len[i], i)
            for i in range(branch_num):
                if i not in [longest_branch.get_min_index(), second_longest_branch.get_min_index()]:
                    structure_node_delete = self.structure_node_list[branch_index[i] // SKELETONIZE.DEGREE_MAX]
                    edge_delete = branch_index[i] % 5
                    self.graph_structure.delete_edge(structure_node_delete.edges[edge_delete])
            self.graph_structure.check_structure()
        
        return changed

    def structure_node_statistic(self, special_node):
        special_node_num = 0
        loopback_count = 0

        for i in range(self.structure_node_num):
            if self.structure_node_list[i].degree > 0:
                if special_node_num == 0:
                    special_node[0] = self.structure_node_list[i]
                else:
                    special_node[1] = self.structure_node_list[i]
                special_node_num += 1
            if self.structure_node_list[i].degree > 1:
                loopback_count += 1

        return special_node_num, loopback_count


    def delete_smaller_loopback(self, bifurcate_node_num, special_node):
        edge_index = [None, None]
        edge_len = [None, None]
        for i in range(2):
            for j in range(special_node[i].degree - 1):
                if special_node[i].edges[j][0] == special_node[i].edges[j][-1]:
                    edge_index[i] = j
                    edge_len[i] = len(special_node[i].edges[j])

        if edge_len[0] > edge_len[1]:
            self.graph_structure.delete_edge(special_node[1].edges[edge_index[1]])
        else:
            self.graph_structure.delete_edge(special_node[0].edges[edge_index[0]])

        bifurcate_node_num -= 1
        return bifurcate_node_num


    def connect_correct_loopback_to_route(self, last_backbone, worm_full_width, special_node, route, change):
        loopback = [[], []]
        loop_count = 0
        for j in range(special_node[1].degree):
            if special_node[1].edges[j][0] == special_node[1].edges[j][-1]:
                loopback[loop_count] = special_node[1].edges[j]
                loop_count += 1

        route.pop()
        forward_route = route + loopback[0]
        backward_route = route + loopback[1]

        last_start = 0
        last_end = last_backbone.length * len(loopback[0]) // (len(loopback[0]) + len(special_node[0].edges[0]))
        if self.same_direction(route, last_backbone):
            last_start = last_backbone.length - 1
            last_end = last_start - last_end + 1

        same_clockwise = self.same_clockwise(loopback[0], last_backbone, last_start, last_end)
        if change:
            same_clockwise = not same_clockwise

        if not same_clockwise:
            route.extend(loopback[0])
        else:
            route.extend(loopback[1])

        bifurcate_cood = self.graph.get_node(loopback[0][0]).center
        while point_dist_square(bifurcate_cood, self.graph.get_node(route[-1]).center) < (worm_full_width ** 2) / 3:
            route.pop()

    def save_structure_nodes(self, special_node):
        with open(".\\cache_data\\structre_node", "wb") as file:
            edge_num = 0
            for n in range(2):
                for i in range(4):
                    if len(special_node[n].edges[i]) > 0:
                        edge_num += 1
                file.write(edge_num.to_bytes(4, 'little'))
                for i in range(edge_num):
                    edge_size = len(special_node[n].edges[i])
                    file.write(edge_size.to_bytes(4, 'little'))
                    for j in range(edge_size):
                        file.write(special_node[n].edges[i][j].to_bytes(4, 'little'))
                edge_num = 0

    def prune(self, graph_before_prune, last_backbone, worm_full_width, is_first_pic, pic_num):
        self.wormShape = "Normal"

        self.graph = graph_before_prune
        self.node_num = self.graph.get_node_num()
        self.node_available = [True] * self.node_num

        self.get_largest_subgraph()

        self.structure_node_num = 0
        for i in range(self.node_num):
            if self.node_available[i] and self.graph.get_node(i).degree != 2:
                self.structure_node_num += 1

        self.graph_structure = GraphStructure(self.node_num, self.structure_node_num)
        self.structure_node_list = self.graph_structure.get_node_list()

        first_node = 0
        second_node = 0
        first_node, second_node = self.start_node_locate(first_node, second_node)
        self.graph_structure_analyze(first_node, second_node)
        self.graph_structure.check_structure()
        self.delete_short_route()
        while self.delete_shorter_routes_with_same_end() or self.delete_branch_and_loopback_except_for_two_longest():
            self.wormShape = "Omega"

        special_node = [GraphStructureNode() for _ in range(2)]
        special_node_num, loopback_count = self.structure_node_statistic(special_node)

        if special_node_num != 2:
            self.wormShape = "Circle"
            raise Exception("Prune Error!!! Special Node Num Must Be 2!!!")
        if is_first_pic and loopback_count > 0:
            raise Exception("First Pic Cannot Have Loopback!")

        if loopback_count == 2:
            loopback_count = self.delete_smaller_loopback(loopback_count, special_node)
        if loopback_count == 1 and special_node[0].degree > 1:
            special_node[0], special_node[1] = special_node[1], special_node[0]

        change = False
        route = special_node[0].edges[0]

        if loopback_count > 0:
            self.wormShape = "Delta"
            self.connect_correct_loopback_to_route(last_backbone, worm_full_width, special_node, route, change)

        backbone_state = BACKBONE_STATE.FORWARD
        if not is_first_pic:
            if point_dist_square(self.graph.get_node(route[0]).center, self.graph.get_node(route[-1]).center) < SKELETONIZE.WORM_SPEED ** 2 * worm_full_width ** 2:
                backbone_state = self.head_tail_recognize(route, last_backbone)
            else:
                backbone_state = self.long_distance_condition(route, last_backbone)

        if backbone_state == BACKBONE_STATE.ABANDON:
            raise Exception("Backbone is abandoned")
        elif backbone_state == BACKBONE_STATE.REVERSE:
            route.reverse()

        new_cood = np.zeros((len(route), 2))
        for i in range(len(route)):
            temp_cood = self.graph.get_node(route[i]).center
            new_cood[i] = temp_cood

        last_backbone.cood = new_cood
        last_backbone.length = len(route)
        last_backbone.size = len(route)
        last_backbone.update_worm_length()
        
        del self.node_available
        del self.graph_structure
        
    def near_distance_condition(self, current_route, last_backbone):
        backbone_state = BACKBONE_STATE.FORWARD
        return backbone_state

    def long_distance_condition(self, current_route, last_backbone):
        backbone_state = BACKBONE_STATE.FORWARD
        p0 = self.graph.get_node(current_route[0]).center
        p1 = self.graph.get_node(current_route[-1]).center
        l0 = last_backbone.cood[0]
        l1 = last_backbone.cood[last_backbone.length - 1]

        current_worm_dir = np.array([p0[0] - p1[0], p0[1] - p1[1]])
        last_worm_dir = np.array([l0[0] - l1[0], l0[1] - l1[1]])

        turning_angle = self.calculate_turning_angle(current_worm_dir, last_worm_dir)

        if turning_angle < SKELETONIZE.WORM_TURNING_ANGLE:
            backbone_state = BACKBONE_STATE.FORWARD
        elif turning_angle > (WORM.PI - SKELETONIZE.WORM_TURNING_ANGLE):
            backbone_state = BACKBONE_STATE.REVERSE
        else:
            start_head_dist = np.linalg.norm(np.array(p0) - np.array(last_backbone.cood[0]))
            end_tail_dist = np.linalg.norm(np.array(p1) - np.array(last_backbone.cood[-1]))
            start_tail_dist = np.linalg.norm(np.array(p0) - np.array(last_backbone.cood[-1]))
            end_head_dist = np.linalg.norm(np.array(p1) - np.array(last_backbone.cood[0]))

            condition_flag = 0
            if (start_tail_dist / start_head_dist > SKELETONIZE.FORWARD_DIST_PORTION and
                    end_head_dist / end_tail_dist > SKELETONIZE.FORWARD_DIST_PORTION):
                condition_flag += 1
                backbone_state = BACKBONE_STATE.FORWARD
            if (start_head_dist / start_tail_dist > SKELETONIZE.FORWARD_DIST_PORTION and
                    end_tail_dist / end_head_dist > SKELETONIZE.FORWARD_DIST_PORTION):
                condition_flag += 1
                backbone_state = BACKBONE_STATE.REVERSE
            if condition_flag == 2:
                backbone_state = BACKBONE_STATE.ABANDON

        return backbone_state

    def head_tail_recognize(self, current_route, last_backbone):
        current_reverse_backbone = Backbone(len(current_route))
        current_backbone = Backbone(len(current_route))
        current_reverse_backbone.length = len(current_route)
        current_backbone.length = len(current_route)

        route_len = len(current_route)
        for i in range(route_len):
            current_reverse_backbone.cood[i] = self.graph.get_node(current_route[route_len - 1 - i]).center
            current_backbone.cood[i] = self.graph.get_node(current_route[i]).center

        es = self.calc_angle_curve_es(current_backbone, last_backbone)
        reverse_es = self.calc_angle_curve_es(current_reverse_backbone, last_backbone)

        if es > reverse_es:
            return BACKBONE_STATE.REVERSE
        else:
            return BACKBONE_STATE.FORWARD

    def calc_angle_curve_es(self, current_backbone, last_backbone):
        last_backbone_smooth = last_backbone
        current_backbone_smooth = current_backbone
        self.root_smooth.interpolate_and_equal_divide(last_backbone_smooth, LARGE_SCALE_PARTITION_NUM)
        self.root_smooth.interpolate_and_equal_divide(current_backbone_smooth, LARGE_SCALE_PARTITION_NUM)

        last_angle_curve = np.zeros(LARGE_SCALE_PARTITION_NUM)
        current_angle_curve = np.zeros(LARGE_SCALE_PARTITION_NUM)
        for i in range(LARGE_SCALE_PARTITION_NUM):
            last_angle_curve[i] = math.atan2(
                last_backbone_smooth.cood[i + 1][1] - last_backbone_smooth.cood[i][1],
                last_backbone_smooth.cood[i + 1][0] - last_backbone_smooth.cood[i][0]
            )
            current_angle_curve[i] = math.atan2(
                current_backbone_smooth.cood[i + 1][1] - current_backbone_smooth.cood[i][1],
                current_backbone_smooth.cood[i + 1][0] - current_backbone_smooth.cood[i][0]
            )

        self.unwrap(last_angle_curve, LARGE_SCALE_PARTITION_NUM)
        self.unwrap(current_angle_curve, LARGE_SCALE_PARTITION_NUM)

        es = np.sum((current_angle_curve - last_angle_curve) ** 2)
        _es = np.sum((current_angle_curve - last_angle_curve - 2 * WORM.PI) ** 2)
        es_ = np.sum((current_angle_curve - last_angle_curve + 2 * WORM.PI) ** 2)

        es = min(es, min(_es, es_))

        return es

    def unwrap(self, v, num):
        diff_wrap = np.zeros(num)
        for i in range(1, num):
            diff_wrap[i] = v[i] - v[i - 1]
            diff_wrap[i] = math.atan2(math.sin(diff_wrap[i]), math.cos(diff_wrap[i]))
        for i in range(1, num):
            v[i] = v[i - 1] + diff_wrap[i]


