import networkx as nx
from scipy.stats import wasserstein_distance
from networkx.algorithms.community import modularity, greedy_modularity_communities


def calculate_simrank(graph, C=0.8, max_iter=10, tol=1e-4):
    """
    Calcula o SimRank para um grafo direcionado ou não.
    
    :param graph: Um objeto NetworkX (DiGraph ou Graph).
    :param C: Fator de decaimento (entre 0 e 1).
    :param max_iter: Número máximo de iterações.
    :param tol: Tolerância para convergência.
    :return: Dicionário com pares de nós e suas similaridades.
    """
    nodes = list(graph.nodes())
    simrank = { (u, v): 1.0 if u == v else 0.0 for u in nodes for v in nodes }

    for _ in range(max_iter):
        prev_simrank = simrank.copy()
        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                # Predecessores dos nós u e v
                predecessors_u = list(graph.predecessors(u))
                predecessors_v = list(graph.predecessors(v))

                if not predecessors_u or not predecessors_v:
                    simrank[(u, v)] = 0.0
                else:
                    simrank[(u, v)] = (C / (len(predecessors_u) * len(predecessors_v))) * sum(
                        prev_simrank[(p_u, p_v)] for p_u in predecessors_u for p_v in predecessors_v
                    )

        # Verificar convergência
        diff = sum(abs(simrank[(u, v)] - prev_simrank[(u, v)]) for u in nodes for v in nodes)
        if diff < tol:
            break

    return simrank

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

