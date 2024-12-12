from tqdm.auto import tqdm
import pandas as pd
import os
import gandula
from gandula.export.dataframe import pff_frames_to_dataframe
from gandula.features.pff import add_ball_speed, add_players_speed
import pyarrow.parquet as pq

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
        frames = []
        for game_id in tqdm(self.game_ids, total=len(self.game_ids), desc="Loading frames"):
            if path:
                metadata_df = pd.read_parquet(f"{path}/{game_id}/metadata.parquet")

                metadata_df = self._reduce_frame_rate(metadata_df, target_fps=5, original_fps=30)

                # read only rows with frame_id in metadata_df

                frame_ids = metadata_df["frame_id"].astype(str).unique().tolist()

                filters = [("frame_id", "in", frame_ids)]

                # Read Parquet file with filtering
                table = pq.read_table(f"{path}/{game_id}/players.parquet", filters=filters)

                # Convert to Pandas DataFrame
                players_df = table.to_pandas()

                frames.append((metadata_df, players_df))
                del metadata_df, players_df
                
            else:
                metadata_df, players_df = pff_frames_to_dataframe(
                    gandula.get_frames(
                        self.data_path,
                        game_id,
                    )
                )

                # Reduce frame rate
                #metadata_df, players_df = self._reduce_frame_rate(metadata_df, players_df,target_fps=5)
                
                # TODO: change_pitch_standards
                # TODO: change_play_side
                # TODO: remove_set_pieces
                
                # Add ball and players speed
                players_df = add_ball_speed(players_df)
                players_df = add_players_speed(players_df)

                frames.append((metadata_df, players_df))
        
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

    def _reduce_frame_rate(self, metadata_df, target_fps=5, original_fps=30):
        """
        Reduces the frame rate of the data by selecting the first frame
        of each interval to achieve the target FPS.

        Args:
            target_fps (int): The desired frame rate after reduction.
            original_fps (int): The original frame rate of the data.

        Returns:
            tuple: Reduced metadata and players DataFrames.
        """
        # Calculate the frame interval in terms of frames

        metadata_df['possession_id'] = metadata_df['possession_id'].ffill(limit=2)
        metadata_df['possession_id'] = metadata_df['possession_id'].bfill(limit=2)
        metadata_df['event_id'] = metadata_df['event_id'].ffill(limit=2)
        metadata_df['event_id'] = metadata_df['event_id'].bfill(limit=2)

        reduced_metadata_df = metadata_df.iloc[::6].reset_index(drop=True)

        return reduced_metadata_df

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