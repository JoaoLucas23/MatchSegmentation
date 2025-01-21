import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from .pass_network import create_team_graphs
import pickle

def get_interval_graphs(passes_df, positions_df):

    interval_ids = positions_df["interval_id"].unique()

    # Initialize list for processed data
    graph_list = []

    team_ids = positions_df['team_id'].unique()
    match_id = positions_df['match_id'][0]

    for interval_id in tqdm(interval_ids, desc="Processing intervals", total=len(interval_ids)):
        interval_passes_df = passes_df[passes_df['interval_id'] == interval_id].reset_index(drop=True)
        interval_positions_df = positions_df[positions_df['interval_id'] == interval_id].reset_index(drop=True)
        graphs = create_team_graphs(interval_passes_df, interval_positions_df, interval_id)

        for team in team_ids:
            
            graph = graphs[str(team)]
            graph_list.append({
                'match_id': match_id,
                'interval_id': interval_id,
                'team_id': team,
                'graph': graph
            })

    return graph_list

def save_graphs(match_id, path, graph_list):
    """
    Processa os dados de passes e posições para um jogo específico.
    
    Parâmetros:
    -----------
    match_id: int
        ID do jogo a ser processado.
    """

    filename = f"{path}/{match_id}.pkl"  # ex.: "1234_graphs.pkl"
    with open(filename, "wb") as f:
        pickle.dump(graph_list, f)
    
def load_graphs(match_id, path):
    """
    Carrega os dados de passes e posições para um jogo específico.
    
    Parâmetros:
    -----------
    match_id: int
        ID do jogo a ser carregado.
    """
    
    filename = f"{path}/{match_id}.pkl"  # ex.: "1234_graphs.pkl"
    with open(filename, "rb") as f:
        graph_list = pickle.load(f)
    return graph_list