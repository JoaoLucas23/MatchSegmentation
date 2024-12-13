import pandas as pd
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from src.viz.graph import plot_graph
from src.data.interval_graphs import process_interval

class GraphStreamDataset(Dataset):
    def __init__(self, metadata_df, players_df, load=False, path='data/processed/', match_id=None, interval='frame'):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata DataFrame.
            players_df (pd.DataFrame): Players DataFrame.
            load (bool): Whether to load processed data or process raw data.
            path (str): Path to save/load processed data.
            interval (str): Interval type ('frame', 'possession', or 'n_seconds').
        """
        if not load:
            self.metadata_df = metadata_df
            self.players_df = players_df
            self.interval = interval
        
            self.players_df["frame_id"] = self.players_df["frame_id"].astype(int)
            self.metadata_df["frame_id"] = self.metadata_df["frame_id"].astype(int)
            self.metadata_df["match_id"] = self.metadata_df["match_id"].astype(int)
            self.players_df["match_id"] = self.players_df["match_id"].astype(int)

            self.match_id = self.metadata_df["match_id"].values[0]
        
            self.data_list = self.process_data(interval)
        else:
            self.data_list = self.load_data(path, match_id)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def save(self, path):
        """Save the dataset to disk."""
        torch.save(self, path + f'/{self.match_id}.pt')

    def load_data(self, path, match_id):
        """Load the dataset from disk."""
        dataset = torch.load(path + f'/{self.match_id}.pt')
        return dataset.data_list

    def get_args(self, merged_df, interval, fully_connected):
        """Prepare arguments for multiprocessing based on the interval type."""
        if interval == 'frame':
            frame_ids = merged_df["frame_id"].unique()
            return [
                (frame_id, merged_df[merged_df["frame_id"] == frame_id], fully_connected)
                for frame_id in tqdm(frame_ids, desc="Preparing arguments", total=len(frame_ids))
            ]

        elif interval == 'possession':
            possession_ids = merged_df["possession_id"].unique()
            return [
                (possession_id, merged_df[merged_df["possession_id"] == possession_id], fully_connected)
                for possession_id in tqdm(possession_ids, desc="Preparing arguments", total=len(possession_ids))
            ]

        elif 'n_seconds' in interval:
            n = int(interval.split('_')[0])
            merged_df['interval_id'] = (merged_df['elapsed_seconds'] // n).astype(int)
            interval_ids = merged_df["interval_id"].unique()
            return [
                (interval_id, merged_df[merged_df["interval_id"] == interval_id], fully_connected)
                for interval_id in tqdm(interval_ids, desc="Preparing arguments", total=len(interval_ids))
            ]

        else:
            raise ValueError(f"Unknown interval type: {interval}")


    def process_data(self, interval='frame', fully_connected=False, num_workers=None):
        """
        Processes the raw data and returns a list of PyTorch Data objects.

        Args:
            interval (str): Interval type ('frame', 'possession', or 'n_seconds').
            fully_connected (bool): Whether the graph is fully connected.
            num_workers (int, optional): Number of workers for multiprocessing. Defaults to cpu_count() - 1.

        Returns:
            list: A list of processed data objects.
        """
        merged_df = pd.merge(
            self.players_df,
            self.metadata_df[["frame_id", "match_id", "possession_id", "home_has_possession"]],
            on=["frame_id", "match_id"],
            how="left",
        )

        # Prepare arguments for multiprocessing
        args = self.get_args(merged_df, interval, fully_connected)

        # Default to cpu_count() - 1 if num_workers is not provided
        if num_workers is None:
            num_workers = max(1, cpu_count() - 1)

        # Initialize list for processed data
        data_list = []

        # Use multiprocessing to process data in parallel
        with Pool(processes=num_workers) as pool:
            with tqdm(total=len(args), desc="Processing data") as pbar:
                for graph in pool.imap_unordered(process_interval, args):
                    data_list.append(graph)
                    pbar.update()

        return data_list

    def view(self, idx: int = 0):
        """Visualize a single interval graph."""
        plot_graph(self.data_list[idx])

    def get_graph_stream(self):
        """Return the graph stream."""
        for data in self.data_list:
            yield data