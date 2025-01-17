import numpy as np
import networkx as nx


def sequence_to_graph(args):
    """
    Constrói 2 grafos (um para cada time) a partir de uma 'sequência' (sequence_df).
    Cada nó é um jogador do time, e cada aresta representa um passe ou
    condução de bola (aresta para si mesmo).
    
    Retorna: (G_time1, G_time2)
    """
    sequence_id, sequence_df, events_df = args

    # Identifica quais times estão nessa sequência.
    # Supondo que seja sempre 2 times, mas se houver mais,
    # você pode iterar sobre todos.
    teams = sequence_df['team'].unique()
    
    # Ordena ou apenas garante que temos 2 times.
    # (Se tiver certeza que são sempre 2 times, pode fazer: teamA, teamB = teams)

    team_graphs = []
    
    for team in teams:
        # Filtra a parte do DataFrame apenas para os jogadores desse time
        sub_sequence_df = sequence_df[sequence_df['team'] == team]
        
        # Cria os nós (player_id -> índice interno)
        node_features, node_team, node_id_map = process_nodes(sub_sequence_df)

        # Filtra os eventos para incluir apenas aqueles em que o 'player_id' (passador/portador)
        # está no sub_sequence_df. Assim, consideramos só eventos desse time.
        sub_events_df = events_df[
            (events_df['possession_type'].isin(['PASS','BALL_CARRY']))
        ].copy()

        # Processa edges
        edge_index, edge_attrs = process_edges(sub_events_df, node_id_map)

        # Cria o grafo para este time
        G = nx.DiGraph(name=f"{sequence_id}_{team}_team")

        # Adiciona nós
        for player_id, idx in node_id_map.items():
            G.add_node(
                idx,
                player_id=player_id,
                team_id=node_team[idx],
                features=node_features[idx]
            )

        # Adiciona arestas
        for (source, target), attr in zip(edge_index, edge_attrs):
            G.add_edge(source, target, **attr)

        team_graphs.append(G)
    
    # Se você sabe que são 2 times, pode retornar explicitamente:
    if len(team_graphs) == 2:
        return team_graphs[0], team_graphs[1]
    else:
        # Caso haja 1 ou mais de 2 times, retorne tudo numa lista
        return team_graphs


def process_nodes(df):
    """
    Processa os nós: cada jogador vira um nó.
    Retorna:
    - node_features: Lista com informações (ex.: média x,y) de cada jogador
    - node_team: Lista com o team_id de cada jogador, indexado na mesma ordem
    - node_id_map: Mapeia player_id -> índice interno
    """
    unique_players = df['shirt'].unique()
    node_id_map = {shirt: idx for idx, shirt in enumerate(unique_players)}

    node_team = []
    node_features = []
    
    for shirt in unique_players:
        team_id = df.loc[df['shirt'] == shirt, 'team_id'].iloc[0]
        x_mean = df.loc[df['shirt'] == shirt, 'x'].mean()
        y_mean = df.loc[df['shirt'] == shirt, 'y'].mean()

        node_team.append(team_id)
        node_features.append((x_mean, y_mean))

    return node_features, node_team, node_id_map


def process_edges(df, node_id_map):
    """
    Processa as arestas (passe ou BALL_CARRY).
    - PASS: incrementa contador num_passes
    - BALL_CARRY: incrementa contador num_ball_carries (aresta para si mesmo)
    Retorna:
    - edge_index: lista de (source, target)
    - edge_attrs: lista de dicionários
    """
    edge_dict = {}

    for row in df.itertuples():
        poss_type = row.possession_type
        source = node_id_map[row.player_id]

        if poss_type == 'PASS':
            # Verifica se existe 'pass_recipient_player_id' e se ele
            # também é deste time (ou seja, está no node_id_map).
            if hasattr(row, 'receiver'):
                recipient_id = row.receiver
                if recipient_id in node_id_map:  # só cria aresta se o receptor está no time
                    target = node_id_map[recipient_id]
                    
                    if (source, target) not in edge_dict:
                        edge_dict[(source, target)] = {'num_passes': 0, 'num_ball_carries': 0}
                    edge_dict[(source, target)]['num_passes'] += 1

        elif poss_type == 'BALL_CARRY':
            # Aresta do jogador para ele mesmo
            target = source
            if (source, target) not in edge_dict:
                edge_dict[(source, target)] = {'num_passes': 0, 'num_ball_carries': 0}
            edge_dict[(source, target)]['num_ball_carries'] += 1

    edge_index = []
    edge_attrs = []

    for (source, target), attrs in edge_dict.items():
        edge_index.append((source, target))
        edge_attrs.append(attrs)

    return edge_index, edge_attrs