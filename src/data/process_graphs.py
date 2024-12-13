import numpy as np
import networkx as nx

def interval_to_graph(args):
    """
    Processa um único intervalo e o transforma em um grafo NetworkX.

    Args:
        args: Tuple contendo (interval_id, interval_df, fully_connected).

    Returns:
        G: um grafo NetworkX com atributos de nós e arestas.
        interval_id: o identificador do intervalo.
    """
    interval_id, interval_df, fully_connected = args

    # Processa nós
    node_features, node_team, node_id_map = process_nodes(interval_df)

    # Processa arestas
    node_ids = list(node_id_map.values())
    edge_index, edge_attrs = process_edges(node_features, node_ids, fully_connected)

    # Cria um grafo NetworkX
    # Escolha entre nx.Graph() ou nx.DiGraph() dependendo da natureza das arestas
    G = nx.Graph()
    G.graph['interval_id'] = interval_id

    # Adiciona nós ao grafo
    # Aqui, indexamos pelos node_ids criados anteriormente
    for original_idx, mapped_id in node_id_map.items():
        # Extraia as features se quiser armazená-las como atributos
        # node_features[mapped_id] = [x, y, vx, vy, ball_team]
        x, y, vx, vy, ball_team = node_features[mapped_id]

        # Você pode adicionar outros atributos do interval_df se necessário
        team = node_team[mapped_id]

        G.add_node(mapped_id, 
                   x=x, 
                   y=y, 
                   vx=vx, 
                   vy=vy, 
                   ball_team=ball_team, 
                   team=team)

    # Adiciona arestas ao grafo
    for (i, j), attr in zip(edge_index, edge_attrs):
        distance = attr[0]
        G.add_edge(i, j, distance=distance)

    return G, interval_id

def calculate_distance(x1, y1, x2, y2):
    """Calcula distância Euclidiana entre dois pontos."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def process_nodes(interval_df):
    """Processa os nós e retorna as features, time e mapeamento de IDs."""
    node_features = []
    node_team = []
    node_id_map = {}

    for idx, row in interval_df.iterrows():
        x, y = row["x"], row["y"]
        vx, vy = row["vx"], row["vy"]
        ball_team = 1 if (row["home_has_possession"] and row["team"] == "home") or (
            not row["home_has_possession"] and row["team"] == "away") else 0

        node_features.append([x, y, vx, vy, ball_team])
        node_team.append(row["team"])
        node_id_map[idx] = len(node_id_map)

    return node_features, node_team, node_id_map

def process_edges(node_features, node_ids, fully_connected):
    """Processa as arestas e retorna índices e atributos."""
    edge_index = []
    edge_attrs = []

    for i in node_ids:
        for j in node_ids:
            if i != j:
                # Verifica se o grafo é fully_connected ou apenas nós do mesmo time
                if fully_connected or node_features[i][4] == node_features[j][4]:
                    xi, yi = node_features[i][0], node_features[i][1]
                    xj, yj = node_features[j][0], node_features[j][1]
                    distance = calculate_distance(xi, yi, xj, yj)
                    edge_index.append((i, j))
                    edge_attrs.append([distance])

    return edge_index, edge_attrs
