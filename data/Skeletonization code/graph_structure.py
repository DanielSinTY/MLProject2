from constant import SKELETONIZE

class GraphStructureNode:
    def __init__(self):
        self.degree = 0
        self.edges = [[] for _ in range(SKELETONIZE.DEGREE_MAX)]

class GraphStructure:
    def __init__(self, real_node_num, structure_node_max):
        self.nodes = [GraphStructureNode() for _ in range(structure_node_max)]
        self.node_hash = [-1] * real_node_num
        self.node_num = 0

    def check_structure(self):
        for node_index in range(self.node_num):
            structure_node_0 = self.nodes[node_index]

            if structure_node_0.degree != 2:
                continue

            adjacents = [None, None]
            edge_index = [None, None]

            for i in range(2):
                adjacent_node_index = self.node_hash[structure_node_0.edges[i][-1]]
                adjacents[i] = self.nodes[adjacent_node_index]

                for j in range(adjacents[i].degree):
                    if adjacents[i].edges[j][1] == structure_node_0.edges[i][-2]:
                        edge_index[i] = j

            adjacents[0].edges[edge_index[0]].pop()
            adjacents[0].edges[edge_index[0]].extend(structure_node_0.edges[1])

            adjacents[1].edges[edge_index[1]].clear()
            adjacents[1].edges[edge_index[1]].extend(reversed(adjacents[0].edges[edge_index[0]]))

            structure_node_0.degree = 0
            self.node_hash[structure_node_0.edges[0][0]] = -1

    def add_edge(self, edge):
        start, end = edge[0], edge[-1]
        if self.node_hash[start] == -1:
            self.node_hash[start] = self.node_num
            self.node_num += 1
        if self.node_hash[end] == -1:
            self.node_hash[end] = self.node_num
            self.node_num += 1

        start_node = self.nodes[self.node_hash[start]]
        end_node = self.nodes[self.node_hash[end]]
        start_node.edges[start_node.degree].extend(edge)
        start_node.degree += 1
        end_node.edges[end_node.degree].extend(reversed(edge))
        end_node.degree += 1

    def delete_edge(self, edge):
        self.delete_edge_oneway(edge[0], edge[1])
        self.delete_edge_oneway(edge[-1], edge[-2])

    def delete_edge_oneway(self, edge_start, midway1):
        if self.node_hash[edge_start] == -1:
            return
        node = self.nodes[self.node_hash[edge_start]]
        for i in range(node.degree):
            if node.edges[i][1] == midway1:
                node.edges[i] = node.edges[node.degree - 1]
                node.degree -= 1
                break

    def move_to_other_end(self, last_node, current_node):
        if self.node_hash[last_node] == -1:
            return False, last_node, current_node
        node = self.nodes[self.node_hash[last_node]]
        for edge in node.edges:
            if not edge:
                break
            if edge[1] == current_node:
                last_node, current_node = edge[-2], edge[-1]
                return True, last_node, current_node
        return False, last_node, current_node

    
    def get_node_list(self):
        return self.nodes

    def get_node_num(self):
        return self.node_num
