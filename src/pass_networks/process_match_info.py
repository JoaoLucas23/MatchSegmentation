import pandas as pd
import numpy as np

def get_match_info(path):
    players_info = pd.read_csv(path+'/players_matches.csv')
    teams_info = pd.read_csv(path+'/teams.csv')
    games_info = pd.read_csv(path+'/games.csv')

    return players_info, teams_info, games_info

def process_players(players_df, match_info, players_info, metadata_events_df):

    players_df = players_df.merge(
        match_info[['match_id', 'home_team_id', 'away_team_id']],
        on='match_id',
        how='left'
    )

    if match_info['home_team_start_left'][0]:
        players_df.loc[players_df['period']==2,'x'] *= -1
        players_df.loc[players_df['period']==2,'y'] *= -1
    else:
        players_df.loc[players_df['period']==1,'x'] *= -1
        players_df.loc[players_df['period']==1,'y'] *= -1

    players_df['team_id'] = np.where(
        players_df['team'] == 'home',
        players_df['home_team_id'],
        players_df['away_team_id']
    )
    players_df.drop(['home_team_id', 'away_team_id'], axis=1, inplace=True)

    players_df['frame_id'] = players_df['frame_id'].astype(float)

    players_df = players_df.merge(metadata_events_df[['match_id','frame_id','interval_id']], on=['match_id','frame_id'], how='left')

    players_df = players_df.merge(players_info, left_on=['match_id','team_id','shirt'], right_on=['match_id','team_id','shirt_number'], how='left').drop_duplicates().reset_index(drop=True)

    return players_df