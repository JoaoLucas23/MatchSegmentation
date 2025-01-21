import networkx as nx
from scipy.stats import wasserstein_distance
from networkx.algorithms.community import modularity, greedy_modularity_communities
import numpy as np
from tqdm.auto import tqdm

def calculate_simrank(graph, C=0.9, max_iter=250, tol=1e-5):
    """
    Calcula o SimRank para um grafo direcionado ou não.
    
    :param graph: Um objeto NetworkX (DiGraph ou Graph).
    :param C: Fator de decaimento (entre 0 e 1).
    :param max_iter: Número máximo de iterações.
    :param tol: Tolerância para convergência.
    :return: Dicionário com pares de nós e suas similaridades.
    """
    #nodes = list(graph.nodes())
    simrank = nx.simrank_similarity(graph, importance_factor=C, max_iterations=max_iter, tolerance=tol)
    nodes = list(graph.nodes())
    n = len(nodes)
    sim_matrix = np.zeros((n, n))

    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            sim_matrix[i, j] = simrank[u][v]

    return np.mean(sim_matrix)


def calculate_wasserstein_distance(G1, G2):

    # Obter distribuições de grau
    degrees_G1 = [deg for _, deg in G1.degree()]
    degrees_G2 = [deg for _, deg in G2.degree()]

    # Calcular similaridade com a distância de Wasserstein (Earth Mover's Distance)
    return wasserstein_distance(degrees_G1, degrees_G2)

def calculate_average_path_legth_target(graph, target=-1):
    """
    Calcula o caminho médio entre todos os nós e um nó alvo.
    
    :param graph: Um objeto NetworkX (DiGraph ou Graph).
    :param target: Nó alvo.
    :return: Caminho médio entre todos os nós e o nó alvo.
    """

    # Calcular o caminho mais curto de todos os nós para o nó-alvo
    shortest_paths = []
    for node in graph.nodes:
        try:
            path_length = nx.shortest_path_length(graph, source=node, target=target)
            shortest_paths.append(path_length)
        except nx.NetworkXNoPath:
            # Caso não haja caminho, ignore (ou use um valor especial, como float('inf'))
            continue

    # Calcular a média dos caminhos (ignorar os sem caminho)
    if shortest_paths:
        return sum(shortest_paths) / len(shortest_paths)
    else:
        return 0.0
    
def calculate_modularity(G):
    communities = list(greedy_modularity_communities(G))
    return modularity(G, communities)

def calculate_ffl(G):
    ffl_count = 0
    for node in G.nodes:
        # Encontrar sucessores e predecessores
        successors = set(G.successors(node))
        for succ in successors:
            second_order_succ = set(G.successors(succ))
            # Contar casos onde o terceiro nó fecha o FFL
            ffl_count += len(second_order_succ & successors)
    return ffl_count

def calculate_graph_distance(G1,G2, method='sum'):

    if method == 'sum':
        return nx.graph_edit_distance(G1, G2) + (1-calculate_simrank(G1, G2)) + calculate_wasserstein_distance(G1, G2)
    elif method == 'avg':
        return np.average([nx.graph_edit_distance(G1, G2), calculate_simrank(G1, G2), calculate_wasserstein_distance(G1, G2)])
    elif method == 'max':
        return max([nx.graph_edit_distance(G1, G2), calculate_simrank(G1, G2), calculate_wasserstein_distance(G1, G2)])
    elif method == 'GED':
        return nx.graph_edit_distance(G1, G2)
    elif method == 'SimRank':
        return 1-calculate_simrank(G1, G2)
    elif method == 'Wasserstein':
        return calculate_wasserstein_distance(G1, G2)
    else:
        raise ValueError("Método inválido. Escolha entre 'sum', 'avg', 'max', 'GED', 'SimRank' ou 'Wasserstein'.")

def calculate_graph_distance_stream(graphs, method='sum'):
    """
    Calcula a distância entre os grafos em uma sequência.
    
    :param graphs: Uma lista de grafos (nx.Graph ou nx.DiGraph).
    :param method: Método para calcular a distância entre os grafos.
                   Opções: 'sum', 'avg', 'max', 'GED', 'SimRank', 'Wasserstein'.
    :return: Uma lista com as distâncias entre os grafos consecutivos.
    """
    distances = []
    for i in tqdm(range(len(graphs)-1), desc=f'Calculando {method}'):
        G1 = graphs[i]
        G2 = graphs[i+1]
        distances.append(calculate_graph_distance(G1, G2, method))
    return distances