import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from .pass_network import sequence_to_graph

def process_sequences(match_dfs: tuple) -> list:

    metadata_df, players_df, events_df = match_dfs

    events_df['possession_id'] = events_df['possession_id'].astype(float)
    metadata_df['possession_id'] = metadata_df['possession_id'].astype(float)

    # Concatenate all match dataframes
    metadata_events = metadata_df.merge(events_df, on="possession_id", how="left")

    metadata_df['frame_id'] = metadata_df['frame_id'].astype(float)
    players_df['frame_id'] = players_df['frame_id'].astype(float)

    df = players_df.merge(metadata_df, on="frame_id", how="left")

    sequences_ids = df["sequence"].unique()
    args = [
        (sequences_id, df[df["sequence"] == sequences_id], metadata_events[metadata_events['sequence'] == sequences_id])
        for sequences_id in tqdm(sequences_ids, desc="Preparing sequences", total=len(sequences_ids))
    ]

    num_workers = max(1, cpu_count() - 2)

    # Initialize list for processed data
    graph_list = []

    # Use multiprocessing to process data in parallel
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(args), desc="Processing data") as pbar:
            for graph in pool.imap_unordered(sequence_to_graph, args):
                graph_list.append(graph)
                pbar.update()

    return graph_list
