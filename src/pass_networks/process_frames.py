"""Exports data to pandas DataFrame."""

from typing import Any

from pandas import DataFrame, json_normalize, merge

from ..providers.pff.schema.tracking import PFF_Frame
from ..schemas.frame.schema import GandulaFrame
from ..visualization.view import pydantic_dump_options


def _build_df(obj_list: list, dump_options: dict[str, Any] | None = None) -> DataFrame:
    if dump_options is None:
        dump_options = {}
    options = {**pydantic_dump_options, **dump_options}
    data = [obj.model_dump(**options) for obj in obj_list]
    return json_normalize(data, sep='_')


def _extract_players_ball_from_frames(
    frames: list[PFF_Frame], *, smoothed: bool = False
) -> list[dict[str, Any]]:
    ball_key = 'ball_with_kalman' if smoothed else 'ball'
    home_key = 'home_players_with_kalman' if smoothed else 'home_players'
    away_key = 'away_players_with_kalman' if smoothed else 'away_players'

    player_dump_options = {'exclude': {'shirt_confidence', 'visibility'}}
    ball_dump_options = {'exclude': {'visibility'}}

    extracted_frames = []

    for frame in frames:
        home_players = getattr(frame, home_key, [])
        away_players = getattr(frame, away_key, [])

        home = [
            {**player.model_dump(**player_dump_options), 'team': 'home'}
            for player in home_players
        ]
        away = [
            {**player.model_dump(**player_dump_options), 'team': 'away'}
            for player in away_players
        ]

        ball = getattr(frame, ball_key, None)
        if ball:
            if isinstance(ball, list):
                ball = ball[0]
            ball_data = [ball.model_dump(**ball_dump_options)]
        else:
            ball_data = []

        players = home + away
        id = frame.frame_id

        extracted_frames.append({'frame_id': id, 'players': players, 'ball': ball_data})

    return extracted_frames


def _build_players_ball_df(
    frames: list[PFF_Frame], *, smoothed: bool = False
) -> DataFrame:
    dump_options = {'include': {'match_id', 'frame_id', 'period', 'elapsed_seconds'}}

    frame_df = _build_df(frames, dump_options)

    coordinates = _extract_players_ball_from_frames(frames, smoothed=smoothed)
    players_df = json_normalize(
        data=coordinates,
        record_path='players',
        meta=['frame_id'],
        sep='_',
    )
    ball_df = json_normalize(
        data=coordinates,
        record_path='ball',
        record_prefix='ball_',
        meta=['frame_id'],
        sep='_',
    )

    players_ball_df = merge(players_df, ball_df, on='frame_id')
    return frame_df.merge(players_ball_df, on='frame_id')


def _build_metadata_df(frames: list[PFF_Frame]) -> DataFrame:
    dump_options = {
        'exclude': {
            'ball',
            'ball_with_kalman',
            'home_players',
            'away_players',
            'home_players_with_kalman',
            'away_players_with_kalman',
        },
        'exclude_unset': False,
        'exclude_defaults': False,
        'exclude_none': False,
    }

    df = _build_df(frames, dump_options)

    # Define the final columns
    final_columns = [
        'match_id',
        'frame_id',
        'period',
        'elapsed_seconds',
        'home_ball',
        'event_id',
        'event_game_event_type',
        'event_setpiece_type',
        'event_player_id',
        'event_team_id',
        'event_start_frame',
        'event_end_frame',
        'possession_id',
        'possession_possession_event_type',
        'possession_start_frame',
        'possession_end_frame',
        'sequence',
        'version',
        'video_time_milli',
    ]

    df = df.loc[:, final_columns]

    # Rename columns
    columns_to_rename = {
        'home_ball': 'home_has_possession',
        'event_game_event_type': 'event_type',
        'possession_possession_event_type': 'possession_type',
    }
    df = df.rename(columns=columns_to_rename)

    return df


def pff_frames_to_dataframe(
    frames: list[PFF_Frame] | list[GandulaFrame], **kwargs
) -> tuple[DataFrame, DataFrame]:
    """Splits PFF frames into metadata and player+ball dfs.

    The PFF Frames collection is transformed into a DataFrame that contains
    frame metadata and another DataFrame with players and ball coordinates.

    Parameters:
    - frames: List of PFF_Frame objects.

    Returns:
    - A tuple containing the metadata DataFrame and the players plus ball DataFrame.
    """
    pitch_size = None
    pitch_center = None
    if isinstance(frames[0], GandulaFrame):
        pitch_size = frames[0].pitch_size
        pitch_center = frames[0].pitch_center
        frames = [frame.frame for frame in frames]

    metadata_df = _build_metadata_df(frames)
    players_ball_df = _build_players_ball_df(frames)

    if pitch_size is not None and pitch_center is not None:
        metadata_df['pitch_size'] = [pitch_size] * len(metadata_df)
        metadata_df['pitch_center'] = [pitch_center] * len(metadata_df)

    return metadata_df, players_ball_df
