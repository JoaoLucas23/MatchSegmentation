import pyarrow.parquet as pq
import pandas as pd

def process_game(args):
    """Process a single game."""
    game_id, path = args

    try:
        metadata_df = pd.read_parquet(f"{path}/{game_id}/metadata.parquet")
        
        # Reduce frame rate
        metadata_df_reduced = _reduce_frame_rate(metadata_df, target_fps=5, original_fps=30)

        # Fill frame ranges
        #metadata_df_reduced = _fill_frame_ranges(metadata_df_reduced)

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
    
def _reduce_frame_rate(metadata_df, target_fps=5, original_fps=30):
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

    print(f"Events: {metadata_df['event_id'].nunique()}")
    print(f"Possessions: {metadata_df['possession_id'].nunique()}")
    
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

    print(f"Reduced events: {reduced_metadata_df['event_id'].nunique()}")
    print(f"Reduced possessions: {reduced_metadata_df['possession_id'].nunique()}")

    return reduced_metadata_df
    
# def _fill_frame_ranges(df: pd.DataFrame) -> pd.DataFrame:
#     # DataFrame de Eventos
#     df_events = df.dropna(subset=['event_id']).reset_index(drop=True)
#     df_possession = df.dropna(subset=['possession_id']).reset_index(drop=True)

#     for _, event in df_events.iterrows():
#         df.loc[(df['frame_id'] >= event['event_start_frame']) & (df['frame_id'] <= event['event_end_frame']), 'event_id'] = event['event_id']
#         df.loc[(df['frame_id'] >= event['event_start_frame']) & (df['frame_id'] <= event['event_end_frame']), 'event_type'] = event['event_type']

#     for _, possession in df_possession.iterrows():
#         df.loc[(df['frame_id'] >= possession['possession_start_frame']) & (df['frame_id'] <= possession['possession_end_frame']), 'possession_id'] = possession['possession_id']
#         df.loc[(df['frame_id'] >= possession['possession_start_frame']) & (df['frame_id'] <= possession['possession_end_frame']), 'possession_type'] = possession['possession_type']

#     return df