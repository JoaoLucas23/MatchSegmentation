import numpy as np
import networkx as nx

def possession_to_graph(args):
    """
    Processa um único intervalo e o transforma em um grafo NetworkX.

    Args:
        args: Tuple contendo (interval_id, interval_df, fully_connected).

    Returns:
        G: um grafo NetworkX com atributos de nós e arestas.
        interval_id: o identificador do intervalo.
    """
    possession_id, possession_df = args


def calculate_distance(x1, y1, x2, y2):
    """Calcula distância Euclidiana entre dois pontos."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def process_nodes(interval_df):
    """Processa os nós e retorna as features, time e mapeamento de IDs."""
    node_features = []
    node_team = []
    node_id_map = {}

    return node_features, node_team, node_id_map

def process_edges(node_features, node_ids, fully_connected):
    """Processa as arestas e retorna índices e atributos."""
    edge_index = []
    edge_attrs = []


