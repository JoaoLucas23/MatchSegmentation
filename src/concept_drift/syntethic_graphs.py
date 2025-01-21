import networkx as nx
from ..pass_networks.custom_metrics import calculate_simrank, calculate_wasserstein_distance

def generate_synthetic_graphs_with_drift(num_graphs, num_nodes, probs_before_drift, probs_after_drift, drift_points):
    graphs = []
    current_prob = probs_before_drift[0]
    drift_index = 0

    for i in range(num_graphs):
        if drift_index < len(drift_points) and i == drift_points[drift_index]:
            current_prob = probs_after_drift[drift_index]
            drift_index += 1

        G = nx.erdos_renyi_graph(num_nodes, current_prob, directed=True)
        graphs.append(G)

    return get_graphs_metrics(graphs)

def get_graphs_metrics(graphs):
    return [calculate_syntenic_graph_metric(graphs[i], graphs[i+1]) for i in range(len(graphs)-1)]

def calculate_syntenic_graph_metric(G1,G2):
    # Calculate the graph edit distance between two graphs

    # np.average([nx.graph_edit_distance(G1, G2), calculate_simrank(G1, G2), calculate_wasserstein_distance(G1, G2)])
    # max([nx.graph_edit_distance(G1, G2), calculate_simrank(G1, G2), calculate_wasserstein_distance(G1, G2)])

    return nx.graph_edit_distance(G1, G2) + (1-calculate_simrank(G1, G2)) + calculate_wasserstein_distance(G1, G2)


