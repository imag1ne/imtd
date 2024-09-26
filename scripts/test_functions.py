import pm4py
import networkx as nx
from imtd.algo.analysis.dfg_functions import edge_case_id_mapping, add_SE, possible_partitions_between_sets, drop_SE, \
    r_to_s


def find_possible_partitions(graph):
    def adj(node_set, graph):
        """
        all adjacent nodes of the node in the node_set
        :param node_set:
        :param graph:
        :return:
        """
        adj_set = set()
        for node in node_set:
            adj_set.update(graph.neighbors(node))
        return adj_set

    activity_list = set(graph.nodes) - {'start', 'end'}

    queue = [(frozenset(), {'start'})]
    visited = set()
    valid = []
    while len(queue) != 0:
        current_state, current_adj = queue.pop()
        for x in current_adj:
            new_state = {x}.union(current_state)
            new_state = frozenset(add_SE(graph, new_state))

            if new_state not in visited:
                visited.add(new_state)
                new_adj = current_adj.union(adj({x}, graph)) - new_state
                queue.append((new_state, new_adj))

                remaining_activities = activity_list - new_state
                if (len(remaining_activities) == 0) or (len(remaining_activities) == len(activity_list)):
                    continue

                remaining_nodes = frozenset(add_SE(graph, remaining_activities))
                graph_remaining_nodes = graph.subgraph(remaining_nodes)

                if 'end' in remaining_nodes:
                    disconnected_nodes_to_end_in_graph_part_b = set(graph_remaining_nodes.nodes) - set(
                        nx.ancestors(graph_remaining_nodes, 'end')) - {'end'}
                    if len(disconnected_nodes_to_end_in_graph_part_b) == 0:
                        valid.append((
                            new_state,
                            remaining_nodes,
                            possible_partitions_between_sets(new_state, remaining_nodes, visited)))
                    else:
                        part_a = frozenset(new_state.union(disconnected_nodes_to_end_in_graph_part_b))
                        if part_a not in visited:
                            visited.add(part_a)
                            part_b = frozenset(remaining_nodes - disconnected_nodes_to_end_in_graph_part_b)
                            graph_part_a = graph.subgraph(part_a)
                            if len(drop_SE(part_b)) != 0 and r_to_s(graph_part_a):
                                valid.append((
                                    part_a,
                                    part_b,
                                    possible_partitions_between_sets(part_a, part_b, visited)))
                            queue.append(
                                (part_a, new_adj.union(adj(disconnected_nodes_to_end_in_graph_part_b, graph)) - part_a))

                if ('end' not in graph_remaining_nodes) and ('start' not in graph_remaining_nodes):
                    if nx.is_weakly_connected(graph_remaining_nodes):
                        valid.append((new_state, remaining_nodes, {"loop"}))

    return valid

def main():
    find_possible_partitions(graph)


if __name__ == '__main__':
    main()