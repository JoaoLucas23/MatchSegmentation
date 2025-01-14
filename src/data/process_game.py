import pyarrow.parquet as pq
import pandas as pd
import numpy as np

import gandula
from gandula.export.dataframe import pff_frames_to_dataframe
from gandula.features.pff import add_ball_speed, add_players_speed

def load_game(args):
    """Process a single game."""
    game_id, path = args

    try:
        metadata_df = pd.read_parquet(f"{path}/{game_id}/metadata.parquet")
        
        # Reduce frame rate
        metadata_df_reduced = reduce_frame_rate(metadata_df, target_fps=5, original_fps=30)

        metadata_df_reduced = filter_invalid_frames(metadata_df)
        metadata_df_reduced = remove_set_pieces(metadata_df_reduced)

        # Get unique frame IDs
        frame_ids = metadata_df_reduced["frame_id"].astype(str).unique().tolist()

        filters = [("frame_id", "in", frame_ids)]

        # Read Parquet file with filtering
        table = pq.read_table(f"{path}/{game_id}/players.parquet", filters=filters)

        # Convert to Pandas DataFrame
        players_df = table.to_pandas()

        return metadata_df, metadata_df_reduced, players_df
    
    except Exception as e:
        return f"Error processing game_id {game_id}: {e}", None, None
    
def process_game(args):

    data_path, game_id = args

    metadata_df, players_df = pff_frames_to_dataframe(
        gandula.get_frames(
            data_path,
            game_id,
        )
    )

    metadata_df_reduced = filter_invalid_frames(metadata_df)
    metadata_df_reduced = remove_set_pieces(metadata_df_reduced)

    metadata_df_reduced = reduce_frame_rate(metadata_df_reduced, target_fps=5, original_fps=30)
    
    # Add ball and players speed
    players_df = add_ball_speed(players_df)
    players_df = add_players_speed(players_df)

    return metadata_df, metadata_df_reduced, players_df
    
def reduce_frame_rate(metadata_df, target_fps=5, original_fps=30):
    """
    Reduces the frame rate of the data by selecting the first frame
    of each interval to achieve the target FPS.

    Args:
        target_fps (int): The desired frame rate after reduction.
        original_fps (int): The original frame rate of the data.

    Returns:
        tuple: Reduced metadata and players DataFrames.
    """
    # Ensure the DataFrame is sorted by the relevant index (e.g., timestamp or frame)
    metadata_df = metadata_df.sort_values('frame_id').reset_index(drop=True)

    metadata_df['frame_id'] = metadata_df['frame_id'].astype(float)

    
    # Identify rows where event_id or possession_id is not null
    key_rows = metadata_df[
        ((metadata_df['event_id'].notna()) & (metadata_df['event_start_frame']==metadata_df['frame_id'])) |
        ((metadata_df['possession_id'].notna()) & (metadata_df['possession_start_frame']==metadata_df['frame_id']))
    ]
    
    # Identify rows where both event_id and possession_id are null
    null_rows = metadata_df[
        (metadata_df['event_id'].isna() | metadata_df['event_start_frame']!=metadata_df['frame_id']) &
        (metadata_df['possession_id'].isna() | metadata_df['possession_start_frame']!=metadata_df['frame_id'])
    ]
    
    step = original_fps // target_fps

    # Downsample null rows to 5 FPS (keep every 6th row)
    downsampled_null_rows = null_rows.iloc[::step]
    
    # Combine key rows and downsampled null rows
    reduced_metadata_df = pd.concat([key_rows, downsampled_null_rows]).sort_index()

    reduced_metadata_df.drop_duplicates(subset='frame_id', keep='first', inplace=True)

    return reduced_metadata_df
    
def filter_invalid_frames(df):
    """
    Filters out invalid frames, keeping rows with valid possession, event, 
    or specific event types (e.g., ON_THE_BALL).

    Args:
        df (pd.DataFrame): Input DataFrame with tracking data.

    Returns:
        pd.DataFrame: Filtered DataFrame with invalid rows removed.
    """
    # Ensure None values are treated as NaN for event and possession IDs
    df['event_id'] = df['event_id'].replace([None], np.nan)
    df['possession_id'] = df['possession_id'].replace([None], np.nan)

    # Define masks
    mask_otb = df['event_type'] == 'ON_THE_BALL'
    mask_valid = (df['home_has_possession'].notna()) & (
        (df['event_id'].notna()) | (df['possession_id'].notna())
    )

    # Combine masks and filter
    filtered_df = df[mask_otb | mask_valid].reset_index(drop=True)
    return filtered_df


def remove_set_pieces(df):
    """
    Removes rows corresponding to set pieces based on the event_setpiece_type column.

    Args:
        df (pd.DataFrame): Input DataFrame with tracking data.

    Returns:
        pd.DataFrame: DataFrame with set piece rows removed.
    """
    # Ensure None values are treated consistently
    df['event_setpiece_type'] = df['event_setpiece_type'].replace(
        ['None', 'nan'], None
    )

    # Filter out rows with non-null set piece types
    filtered_df = df[df['event_setpiece_type'].isnull()].reset_index(drop=True)
    return filtered_df
