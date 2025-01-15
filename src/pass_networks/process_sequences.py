import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from .pass_network import sequences_to_graph

def process_sequences(df):

    sequences_ids = df["sequence"].unique()
    args = [
        (sequences_id, df[df["sequence"] == sequences_id])
        for sequences_id in tqdm(sequences_ids, desc="Preparing sequences", total=len(sequences_ids))
    ]

    num_workers = max(1, cpu_count() - 2)

    # Initialize list for processed data
    graph_list = []

    # Use multiprocessing to process data in parallel
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(args), desc="Processing data") as pbar:
            for graph in pool.imap_unordered(sequences_to_graph, args):
                graph_list.append(graph)
                pbar.update()

    return graph_list
