from tqdm.auto import tqdm
import pandas as pd

import gandula
from gandula.export.dataframe import pff_frames_to_dataframe
from gandula.features.pff import add_ball_speed, add_players_speed

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
                players_df = pd.read_parquet(f"{path}/{game_id}/players.parquet")
                frames.append((metadata_df, players_df))
            else:
                metadata_df, players_df = pff_frames_to_dataframe(
                    gandula.get_frames(
                        self.data_path,
                        game_id,
                    )
                )

                # Reduce frame rate
                metadata_df, players_df = self._reduce_frame_rate(metadata_df, players_df,target_fps=5)
                
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
            
            game_path = f"data/intermediate/{game_ids[i]}"
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

    def _reduce_frame_rate(self, metadata_df, players_df, target_fps=5, original_fps=25):
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

        interval = original_fps / target_fps

        # Validate target frame rate
        if interval < 1:
            raise ValueError("Target FPS must be less than or equal to original FPS.")

        # Select the first frame of each interval
        reduced_metadata_df = metadata_df[
            (metadata_df["frame_id"] // interval).astype(int).diff().fillna(1).astype(bool)
        ]

        # Filter players based on the reduced metadata frame IDs
        reduced_players_df = players_df[
            players_df["frame_id"].isin(reduced_metadata_df["frame_id"])
        ]

        return reduced_metadata_df, reduced_players_df

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