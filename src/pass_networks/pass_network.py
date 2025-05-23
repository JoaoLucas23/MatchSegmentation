import numpy as np
import networkx as nx

from .custom_metrics import calculate_simrank, calculate_wasserstein_distance

def create_team_graphs(passes_df, positions_df, interval_id):
    """
    Cria um grafo dirigido (NetworkX) para cada team_id encontrado em 'passes_df'.
    
    Parâmetros:
    ----------
    passes_df: DataFrame com colunas [match_id, team_id, player_id, receiver_id, count].
               Cada linha representa a contagem de passes de 'player_id' -> 'receiver_id'.
    positions_df: DataFrame com colunas [match_id, team_id, player_id, ball_x, ball_y].
                  Cada linha dá a posição média (x,y) daquele jogador no campo.
    
    Retorno:
    --------
    dict onde a chave é o team_id e o valor é um nx.DiGraph.
    Cada grafo tem:
      - Nós = player_id (com atributo 'pos' e 'features' = (ball_x, ball_y))
      - Arestas dirigidas de player_id -> receiver_id (com 'weight' = count).
    """
    
    passes_df = passes_df[(passes_df['interval_id'] == interval_id)&(passes_df['team_id'].notna())].reset_index(drop=True)
    positions_df = positions_df[(positions_df['interval_id'] == interval_id)&(positions_df['team_id'].notna())].reset_index(drop=True)

    passes_df['team_id'] = passes_df['team_id'].astype(int)
    passes_df['player_shirt'] = passes_df['player_shirt'].astype(int)
    passes_df['receiver_shirt'] = passes_df['receiver_shirt'].astype(int)
    positions_df['team_id'] = positions_df['team_id'].astype(int)


    # Identifica todos os times disponíveis no DF de passes
    teams = positions_df['team_id'].unique()
    graphs_dict = {}

    for team in teams:
        # Filtra o DF de passes para esse time
        sub_passes_df = passes_df[passes_df['team_id'] == team]
        # Filtra o DF de posições para esse time
        sub_positions_df = positions_df[positions_df['team_id'] == team]

        # Cria um grafo dirigido para este time
        G = nx.DiGraph(name=f"{interval_id}_team_{team}")

        # Adiciona cada player_id como nó, com atributo de posição (ball_x, ball_y)
        # Vamos supor que cada combinação (match_id, team_id, player_id) seja única
        for row in sub_positions_df.itertuples():
            player_id = row.player_id
            x_coord =  row.x
            y_coord = row.y
            shirt_number = row.shirt

            # Adicionamos o nó, salvando a posição em 'pos' e 'features'
            G.add_node(
                shirt_number,
                pos=(x_coord, y_coord),
                features=(x_coord, y_coord)
            )

        # Adiciona as arestas a partir de sub_passes_df
        # Direção: player_id -> receiver_id, peso = count
        for row in sub_passes_df.itertuples():
            source = row.player_shirt
            target = row.receiver_shirt
            count = row.count

            # Cria a aresta com peso
            # Se preferir, pode somar caso já exista, mas aqui sobrescrevemos
            G.add_edge(source, target, weight=count)

        # Guarda o grafo no dicionário
        graphs_dict[str(team)] = G
        
    return graphs_dict

def calculate_metrics(graph, metrics):
    """
    Calcula métricas especificadas para um grafo.
    
    :param graph: O grafo a ser analisado (nx.Graph ou nx.DiGraph).
    :param metrics: Um dicionário onde as chaves são os nomes das métricas
                    e os valores são funções que calculam essas métricas.
    :return: Um dicionário com os valores das métricas calculadas.
    """
    results = {}
    for metric_name, metric_function in metrics.items():
        results[metric_name] = metric_function(graph)
    return results
