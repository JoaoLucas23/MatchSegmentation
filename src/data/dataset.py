import pandas as pd
import torch
from torch.utils.data import Dataset
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from src.data.interval import process_interval

class IntervalDataset(Dataset):
    def __init__(self, metadata_df, players_df, load=False, path='data/processed/', interval='frame'):
        """
        Args:
            metadata_df (pd.DataFrame): Metadata DataFrame.
            players_df (pd.DataFrame): Players DataFrame.
            load (bool): Whether to load processed data or process raw data.
            path (str): Path to save/load processed data.
            interval (str): Interval type ('frame', 'possession', or 'n_seconds').
        """
        self.metadata_df = metadata_df
        self.players_df = players_df
        self.interval = interval

        if not load:
            self.data_list = self.process_data(interval)
        else:
            self.data_list = self.load_data(path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

    def save(self, path):
        """Save the dataset to disk."""
        torch.save(self, path)

    def load_data(self, path):
        """Load the dataset from disk."""
        dataset = torch.load(path)
        return dataset.data_list

    def get_args(self, merged_df, interval):
        """Prepare arguments for multiprocessing based on the interval type."""
        if interval == 'frame':
            frame_ids = merged_df["frame_id"].unique()
            return [
                (frame_id, merged_df[merged_df["frame_id"] == frame_id])
                for frame_id in frame_ids
            ]

        elif interval == 'possession':
            possession_ids = merged_df["possession_id"].unique()
            return [
                (possession_id, merged_df[merged_df["possession_id"] == possession_id])
                for possession_id in possession_ids
            ]

        elif 'n_seconds' in interval:
            n = int(interval.split('_')[0])
            merged_df['interval_id'] = (merged_df['elapsed_seconds'] // n).astype(int)
            interval_ids = merged_df["interval_id"].unique()
            return [
                (interval_id, merged_df[merged_df["interval_id"] == interval_id])
                for interval_id in interval_ids
            ]

        else:
            raise ValueError(f"Unknown interval type: {interval}")


    def process_data(self, interval='frame'):
        """
        Processes the raw data and returns a list of PyTorch Data objects.
        
        Args:
            interval (str): Interval type ('frame', 'possession', or 'n_seconds').

        Returns:
            list: A list of processed data objects.
        """
        merged_df = pd.merge(
            self.players_df,
            self.metadata_df[["frame_id", "match_id", "possession_id"]],
            on=["frame_id", "match_id"],
            how="left",
        )

        # Handle missing possession_ids
        merged_df["possession_id"] = merged_df["possession_id"].fillna(-1)

        # Prepare arguments for multiprocessing
        args = self.get_args(merged_df, interval)

        # Initialize list for processed data
        data_list = []

        # Use multiprocessing to process data in parallel
        num_workers = max(1, cpu_count() - 2)  # Leave one CPU free
        with Pool(processes=num_workers) as pool:
            # Use tqdm for progress bar
            with tqdm(total=len(args), desc="Processing data") as pbar:
                for graph in pool.imap_unordered(process_interval, args):
                    data_list.append(graph)
                    pbar.update()

        return data_list
