from constant import SKELETONIZE, BW, WORM
import pickle

class GraphNode:
    DEGREE_MAX = SKELETONIZE.DEGREE_MAX
    INF = WORM.INF

    def __init__(self, center=None, degree=0, adjacent=None, node_from=None, copy_adjacent=True):
        if node_from is not None:
            # Copy constructor
            self.center = node_from.center.copy()
            if copy_adjacent:
                self.degree = node_from.degree
                self.adjacent = node_from.adjacent[:node_from.degree]
            else:
                self.degree = 0
                self.adjacent = [None] * self.DEGREE_MAX
        else:
            # Default constructor
            self.center = [self.INF, self.INF] if center is None else center
            self.degree = degree
            self.adjacent = [None] * self.DEGREE_MAX if adjacent is None else adjacent

    def __getitem__(self, index):
        return self.adjacent[index]

    def __setitem__(self, index, value):
        self.adjacent[index] = value

    def __eq__(self, other):
        if not isinstance(other, GraphNode):
            return False
        return self.center == other.center and self.degree == other.degree and self.adjacent[:self.degree] == other.adjacent[:self.degree]

    def __copy__(self):
        return GraphNode(self, True)


    def select_next(self, last_node, current_node):
        if self.degree != 2:
            return False, last_node, current_node
        next_node = self.adjacent[1] if last_node == self.adjacent[0] else self.adjacent[0]
        last_node, current_node = current_node, next_node
        return True, last_node, current_node

    def get_adjacent_index(self, adjacent_node):
        try:
            return self.adjacent.index(adjacent_node)
        except ValueError:
            return -1

    def __repr__(self):
        return f"GraphNode(center={self.center}, degree={self.degree}, adjacent={self.adjacent[:self.degree]})"

    def __str__(self):
        return self.__repr__()

class Graph:
    POINT_NUM_MAX = SKELETONIZE.POINT_NUM_MAX

    def __init__(self):
        self.node_num = 0
        self.nodes = [GraphNode() for _ in range(self.POINT_NUM_MAX)]

    def reset(self):
        self.node_num = 0

    def get_node_num(self):
        return self.node_num

    def get_node(self, node_index):
        if node_index < 0 or node_index >= self.node_num:
            return None
        return self.nodes[node_index]

    def connect_node(self, node_1, node_2):
        self.nodes[node_1].adjacent[self.nodes[node_1].degree] = node_2
        self.nodes[node_1].degree += 1
        self.nodes[node_2].adjacent[self.nodes[node_2].degree] = node_1
        self.nodes[node_2].degree += 1

    def add_node(self, center, fu_node=-1):
        self.nodes[self.node_num].center = center[:]
        self.nodes[self.node_num].degree = 0
        if fu_node != -1:
            self.nodes[self.node_num].adjacent[self.nodes[self.node_num].degree] = fu_node
            self.nodes[self.node_num].degree += 1
            self.nodes[fu_node].adjacent[self.nodes[fu_node].degree] = self.node_num
            self.nodes[fu_node].degree += 1
        self.node_num += 1

    @staticmethod
    def persistence(obj_ptr, out_file):
        with open(out_file, 'wb') as file:
            pickle.dump(obj_ptr, file)

    @staticmethod
    def anti_persistence(obj_ptr, in_file):
        with open(in_file, 'rb') as file:
            loaded_obj = pickle.load(file)
            obj_ptr.__dict__.update(loaded_obj.__dict__)