import networkx as nx

class Match:
    def __init__(self, match_id, home_team_id, away_team_id, metadata_df, players_df):
        self.match_id = match_id
        self.home_team_id = home_team_id
        self.away_team_id = away_team_id

        # Cria o graph_stream completo (consideramos que GraphStream aceita df_tuple ou path)
        # Aqui assumimos que queremos processar da forma 'frame', ajuste se necessário
        self.graph_stream = GraphStream(df_tuple=(metadata_df, players_df), fully_conected=True)
        
        home_players_df = players_df[players_df['team'] == "home"].reset_index(drop=True)
        away_players_df = players_df[players_df['team'] == "away"].reset_index(drop=True)

        # Inicializa os grafos de cada time
        self.home_stream = GraphStream(df_tuple=(metadata_df, home_players_df), fully_conected=True)
        self.away_stream = GraphStream(df_tuple=(metadata_df, away_players_df), fully_conected=True)


        # Inicializa a lista de métricas
        self.graph_metrics = []
        self.home_metrics = []
        self.away_metrics = []

    def __getitem__(self, idx):
        return self.graph_stream[idx], self.home_stream[idx], self.away_stream[idx]

    def __len__(self):
        return len(self.graph_stream)

    def _get_graph_by_frame_id(self, stream, frame_id):
        """
        Dado um GraphStream (ou lista de tuplas) e um frame_id,
        retorna o grafo correspondente ou None se não encontrado.
        """
        for G, fid in stream.graphs:
            if fid == frame_id:
                return G
        return None

    def get_graphs_by_frame_id(self, frame_id):
        """
        Retorna uma tupla (G_full, G_home, G_away) para um frame_id específico.
        """
        G_full = self._get_graph_by_frame_id(self.graph_stream, frame_id)
        G_home = self._get_graph_by_frame_id(self.home_stream, frame_id)
        G_away = self._get_graph_by_frame_id(self.away_stream, frame_id)
        return G_full, G_home, G_away