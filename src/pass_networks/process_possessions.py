import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
from .pass_network import possession_to_graph

def process_possession(df):

    possession_ids = df["possession_id"].unique()
    args = [
        (possession_id, df[df["possession_id"] == possession_id])
        for possession_id in tqdm(possession_ids, desc="Preparing arguments", total=len(possession_ids))
    ]

    num_workers = max(1, cpu_count() - 2)

    # Initialize list for processed data
    graph_list = []

    # Use multiprocessing to process data in parallel
    with Pool(processes=num_workers) as pool:
        with tqdm(total=len(args), desc="Processing data") as pbar:
            for graph in pool.imap_unordered(possession_to_graph, args):
                graph_list.append(graph)
                pbar.update()

    return graph_list


