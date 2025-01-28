import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import os
import gandula
from gandula.export.dataframe import pff_frames_to_dataframe
from gandula.features.pff import add_ball_speed, add_players_speed
from .process_events import get_match_events

def process_game(args):

    data_path, game_id = args

    events_df = get_match_events(game_id)

    if not isinstance(events_df, pd.DataFrame) or events_df.empty:
        return None, None, None

    metadata_df, players_df = pff_frames_to_dataframe(
        gandula.get_frames(
            data_path,
            game_id,
        )
    )

    

    events_df = events_df.drop_duplicates(subset=['event_id','possession_id']).reset_index(drop=True)

    return process_metadata(metadata_df), players_df, events_df

def load_game(args):
    """Process a single game."""
    path, game_id = args

    try:

        events_df = pd.read_csv(f"{path}/{game_id}/events.csv")

        metadata_df = pd.read_parquet(f"{path}/{game_id}/metadata.parquet")
        
        players_df = pd.read_parquet(f"{path}/{game_id}/players.parquet")

        return metadata_df, players_df, events_df
    
    except Exception as e:
        return f"Error processing game_id {game_id}: {e}", None, None


def save_game(metadata_df,players_df,events_df,path,game_id):
    def make_serializable(df):
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].apply(
                    lambda x: x.name if hasattr(x, "name") else str(x)
                )
        return df

    metadata_df["event_type"] = metadata_df["event_type"].apply(
        lambda x: x.name if hasattr(x, "name") else str(x)
    )
    metadata_df = make_serializable(metadata_df)
    players_df = make_serializable(players_df)
    
    game_path = f"{path}/{game_id}"
    os.makedirs(game_path, exist_ok=True)

    # Save the DataFrames
    metadata_df.to_parquet(f"{game_path}/metadata.parquet", engine="fastparquet")
    players_df.to_parquet(f"{game_path}/players.parquet", engine="fastparquet")
    events_df.to_csv(f"{game_path}/events.csv")

def process_metadata(metadata_df):
    max_seconds = metadata_df.loc[metadata_df['period']==1,'elapsed_seconds'].max()
    metadata_df['seconds'] =  metadata_df['elapsed_seconds'] + (max_seconds * (metadata_df['period']-1))

    metadata_df['interval_id'] =  (metadata_df['seconds']//120 )+ 1
    metadata_df['interval_id'] = metadata_df['interval_id'].astype(int)

    metadata_df['event_setpiece_type'] = metadata_df['event_setpiece_type'].astype(str)

    metadata_events_df = metadata_df[metadata_df['event_setpiece_type'].isin(['SetPieceType.KICK_OFF','SetPieceType.GOAL_KICK','nan', 'None'])]

    metadata_events_df['frame_id'] = metadata_events_df['frame_id'].astype(int)
    metadata_events_df['event_id'] = metadata_events_df['event_id'].astype(float)
    metadata_events_df['event_start_frame'] = metadata_events_df['event_start_frame'].astype(float)
    metadata_events_df['event_end_frame'] = metadata_events_df['event_end_frame'].astype(float)
    metadata_events_df['possession_id'] = metadata_events_df['possession_id'].astype(float)
    metadata_events_df['possession_start_frame'] = metadata_events_df['possession_start_frame'].astype(float)
    metadata_events_df['possession_end_frame'] = metadata_events_df['possession_end_frame'].astype(float)
    metadata_events_df['period'] = metadata_events_df['period'].astype(int)
    metadata_events_df['match_id'] = metadata_events_df['match_id'].astype(int)

    return metadata_events_df

