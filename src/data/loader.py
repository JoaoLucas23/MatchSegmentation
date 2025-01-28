from tqdm.auto import tqdm
import pandas as pd
import os
import gandula

from multiprocessing import Pool
import numpy as np
import pyarrow.parquet as pq

from src.data.process_game import process_game, load_game, filter_invalid_frames, remove_set_pieces, reduce_frame_rate

class FramesLoader:
    def __init__(
        self,
        game_ids: list[int],
        data_path: (
            str
        ) = "data/raw/",
        # remove_set_pieces: bool = True, # TODO: remove set pieces
        # TODO: add filters on numbers of events on a possession
        # TODO: add filters on size of the possession
    ):
        self.game_ids = game_ids
        self.data_path = data_path
        self.frames = []

    def load(self, path: str | None = None):
        """
        """

        if path:
            frames = []

            tasks = [(game_id, path) for game_id in self.game_ids]

            num_workers = 2  # Leave one CPU free
            with Pool(processes=num_workers) as pool:
                # Use tqdm for progress bar
                with tqdm(total=len(tasks), desc="Loading Games") as pbar:
                    for match_id in pool.imap_unordered(load_game, tasks):
                        frames.append(match_id)
                        pbar.update()

                
        else:
            frames = []
            tasks = [(game_id, path) for game_id in self.game_ids]

            num_workers = 2  # Leave one CPU free
            with Pool(processes=num_workers) as pool:
                # Use tqdm for progress bar
                with tqdm(total=len(tasks), desc="Processing Games") as pbar:
                    for match_id in pool.imap_unordered(process_game, tasks):
                        frames.append(match_id)
                        pbar.update()
        
        self.frames = frames

    def get(self) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        return self.frames
    
    def save(self, path: str = 'data/processed/'):
        def make_serializable(df):
            for col in df.columns:
                if df[col].dtype == "object":
                    df[col] = df[col].apply(
                        lambda x: x.name if hasattr(x, "name") else str(x)
                    )
            return df
        for i, (metadata_df, players_df) in tqdm(enumerate(self.frames), total=len(self.frames), desc="Saving frames"):
            metadata_df["event_type"] = metadata_df["event_type"].apply(
                lambda x: x.name if hasattr(x, "name") else str(x)
            )
            metadata_df = make_serializable(metadata_df)
            players_df = make_serializable(players_df)
            
            game_path = f"{path}/{self.game_ids[i]}"
            os.makedirs(game_path, exist_ok=True)

            # Save the DataFrames
            metadata_df.to_parquet(f"{game_path}/metadata.parquet", engine="fastparquet")
            players_df.to_parquet(f"{game_path}/players.parquet", engine="fastparquet")

    def _filter_possessions(
        self, metadata_df: pd.DataFrame, players_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        possession_start_end = (
            metadata_df[["event_start_frame", "event_end_frame"]]
            .dropna()
            .drop_duplicates()
            .values
        )

        # remove start and end frames that are the same
        possession_start_end = possession_start_end[
            possession_start_end[:, 0] != possession_start_end[:, 1]
        ]

        metadata_df = metadata_df[
            metadata_df["frame_id"].isin(
                np.concatenate(
                    [
                        np.arange(start, end + 1)
                        for start, end in possession_start_end.astype(int)
                    ]
                )
            )
        ]

        filtered_players_df = players_df[
            players_df["frame_id"].isin(
                np.concatenate(
                    [
                        np.arange(start, end + 1)
                        for start, end in possession_start_end.astype(int)
                    ]
                )
            )
        ]

        # eliminate possessions with only one frame
        metadata_df["possesion_length"] = (
            metadata_df["event_end_frame"] - metadata_df["event_start_frame"]
        )
        metadata_df = metadata_df[metadata_df["possesion_length"] > 1]

        frames = metadata_df["frame_id"].unique()
        filtered_players_df = filtered_players_df[
            filtered_players_df["frame_id"].isin(frames)
        ]

        return metadata_df, filtered_players_df

    def _remove_set_pieces(
        self, metadata_df: pd.DataFrame, players_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:  # TODO: implement
        pass


    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx]

    def __iter__(self):
        for frame in self.frames:
            yield frame

    def __repr__(self):
        return f"<Frames: {len(self)} frames>"
    


