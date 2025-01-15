import numpy as np
import networkx as nx

def sequence_to_graph(args):
    """
    Constrói um grafo a partir de uma posse de bola (sequence_df).
    Cada nó é um jogador, e cada aresta representa um passe, cruzamento ou
    condução de bola (aresta para si mesmo).
    """
    sequence_id, sequence_df = args

    # Processa nós
    node_features, node_team, node_id_map = process_nodes(sequence_df)
    
    # Filtra apenas as ações que geram arestas na rede
    edges_df = sequence_df[
        sequence_df['possession_type'].isin(['PASS','BALL_CARRY'])
    ]

    # Processa arestas
    edge_index, edge_attrs = process_edges(edges_df, node_id_map)

    # (Opcional) Cria um grafo direcionado (NetworkX) para visualização ou análises
    G = nx.DiGraph(name=f"{sequence_id}")
    
    # Adiciona nós ao grafo
    # Aqui, estamos usando o ID interno (índice) de cada jogador como nó
    # e também salvamos algumas informações extras em atributos de nó.
    for player_id, idx in node_id_map.items():
        G.add_node(
            idx,
            player_id=player_id,
            team_id=node_team[idx],
            features=node_features[idx]
        )

    # Adiciona arestas ao grafo
    # edge_index é uma lista de tuplas (source, target)
    # edge_attrs é a lista de dicionários com atributos das arestas
    for (source, target), attr in zip(edge_index, edge_attrs):
        G.add_edge(source, target, **attr)

    # Retorna o grafo criado (ou, se preferir, retorne edge_index, edge_attrs, etc.)
    return G

def process_nodes(df):
    """
    Processa os nós: cada jogador envolvido na posse de bola vira um nó.
    Retorna:
    - node_features: Lista com informações (features) de cada jogador
    - node_team: Lista com o time de cada jogador no índice correspondente
    - node_id_map: Mapeia player_id -> índice interno do nó
    """
    # Identifica jogadores únicos nessa posse
    unique_players = df['player_id'].unique()

    # Mapeamento player_id -> índice (0,1,2,...)
    node_id_map = {player_id: idx for idx, player_id in enumerate(unique_players)}

    node_team = []
    node_features = []
    
    # Para cada jogador único, podemos armazenar as features de interesse
    for player_id in unique_players:
        # Exemplo: pegar o time do jogador (assumindo que seja consistente no DF)
        team_id = df.loc[df['player_id'] == player_id, 'team_id'].iloc[0]

        # Exemplo de features que poderiam ser salvas:
        x_mean = df.loc[df['player_id'] == player_id, 'x'].mean()
        y_mean = df.loc[df['player_id'] == player_id, 'y'].mean()
        
        # Para simplificar, vamos criar um vetor vazio ou com um valor fixo

        node_team.append(team_id)
        node_features.append((x_mean, y_mean))

    return node_features, node_team, node_id_map

def process_edges(df, node_id_map):
    """
    Processa as arestas com base no DataFrame filtrado (apenas PASS, CROSS, BALL_CARRY).
    Retorna:
    - edge_index: lista de (source, target) no índice interno
    - edge_attrs: lista de dicionários com atributos de cada aresta
    """
    edge_dict = {}

    for row in df.itertuples():
        # row.possession_type pode ser PASS, CROSS ou BALL_CARRY
        poss_type = row.possession_type
        source = node_id_map[row.player_id]
        
        if poss_type == 'PASS':
            # Precisamos ter a informação de quem recebeu o passe/cruzamento
            # Aqui assumimos que existe a coluna 'pass_recipient_player_id'
            # Ajuste conforme seu DataFrame real
            target = node_id_map[row.pass_recipient_player_id]
            
            # Se não existir essa (source,target) no dicionário, inicia
            if (source, target) not in edge_dict:
                edge_dict[(source, target)] = {
                    'num_passes': 0,
                    'num_ball_carries': 0
                }
            
            # Incrementa o contador de passes (ou de "ações")
            edge_dict[(source, target)]['num_passes'] += 1

        elif poss_type == 'BALL_CARRY':
            # Se for condução de bola, aresta do jogador para ele mesmo
            source = node_id_map[row.player_id]
            target = source
            if (source, target) not in edge_dict:
                edge_dict[(source, target)] = {
                    'num_passes': 0,
                    'num_ball_carries': 0
                }
            edge_dict[(source, target)]['num_ball_carries'] += 1

    edge_index = []
    edge_attrs = []

    for (source, target), attrs in edge_dict.items():
        edge_index.append((source, target))
        edge_attrs.append(attrs)

    return edge_index, edge_attrs
