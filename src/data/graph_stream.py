import networkx as nx
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool, cpu_count
import pickle

from src.data.process_graphs import interval_to_graph
from src.viz.graph import plot_graph

class GraphStream:
    def __init__(self, df_tuple: tuple | None = None, fully_conected: bool = False, path: str | None = None):
        if path:
            self.graphs = self._load_graphs(path)
        else:
            self.metadata_df = df_tuple[0]
            self.players_df = df_tuple[1]
            self.graphs = self._create_graphs(fully_connected=False)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def _get_args(self, merged_df, interval, fully_connected):
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

    def _create_graphs(self, interval='frame', fully_connected=False):
        """
        Processes the raw data and returns a list of PyTorch Data objects.

        Args:
            interval (str): Interval type ('frame', 'possession', or 'n_seconds').
            fully_connected (bool): Whether the graph is fully connected.
            num_workers (int, optional): Number of workers for multiprocessing. Defaults to cpu_count() - 1.

        Returns:
            list: A list of processed data objects.
        """
        self.metadata_df["frame_id"] = self.metadata_df["frame_id"].astype(int)
        self.players_df["frame_id"] = self.players_df["frame_id"].astype(int)
        self.metadata_df["match_id"] = self.metadata_df["match_id"].astype(int)
        self.players_df["match_id"] = self.players_df["match_id"].astype(int)

        merged_df = pd.merge(
            self.players_df,
            self.metadata_df[["frame_id", "match_id", "possession_id", "home_has_possession"]],
            on=["frame_id", "match_id"],
            how="left",
        )

        # Prepare arguments for multiprocessing
        args = self._get_args(merged_df, interval, fully_connected)

        # Default to cpu_count() - 1 if num_workers is not provided
        num_workers = max(1, cpu_count() - 2)

        # Initialize list for processed data
        data_list = []

        # Use multiprocessing to process data in parallel
        with Pool(processes=num_workers) as pool:
            with tqdm(total=len(args), desc="Processing data") as pbar:
                for graph in pool.imap_unordered(interval_to_graph, args):
                    data_list.append(graph)
                    pbar.update()

        del merged_df
        return data_list

    def _load_graphs(self, path: str):
        """Load graphs from a given path."""
        with open(path, 'rb') as f:
            graphs = pickle.load(f)
        return graphs

    def save(self, path: str, file_name: str):
        """Save the graph stream to a given path as picle."""

        path = f"{path}/{file_name}.pkl"

        with open(path, 'wb') as f:
            pickle.dump(self.graphs, f)

    def view(self, idx: int | list[int] = 0):
        """Visualize a single interval graph."""

        if isinstance(idx, int):
            plot_graph(self.graphs[idx])
        else:
            selected_graphs = [self.graphs[i] for i in idx]
            plot_graph_sequence(selected_graphs)
            del selected_graphs

    def get_graph_stream(self):
        """Return the graph stream."""
        for data in self.graphs:
            yield data