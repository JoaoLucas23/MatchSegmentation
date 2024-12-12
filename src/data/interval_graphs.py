import torch
from torch_geometric.data import Data
import numpy as np


def process_interval(args):
    """
    Process a single interval and transform it into a PyTorch Geometric graph.

    Args:
        args: Tuple containing (interval_id, interval_df, fully_connected).

    Returns:
        Data: PyTorch Geometric Data object.
    """

    interval_id, interval_df, fully_connected = args

    # Process nodes
    node_features, node_team, node_id_map = process_nodes(interval_df)

    # Process edges
    node_ids = list(node_id_map.values())
    edge_index, edge_attrs = process_edges(node_features, node_ids, fully_connected)

    # Convert to PyTorch tensors
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)

    # Create PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attrs)
    data.interval_id = interval_id

    return data

def calculate_distance(x1, y1, x2, y2):
        """Calculate Euclidean distance between two points."""
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def process_nodes(interval_df):
    """Process nodes and return features, team affiliation, and node mapping."""
    node_features = []
    node_team = []
    node_id_map = {}

    for idx, row in interval_df.iterrows():
        x, y = row["x"], row["y"]
        vx, vy = row["vx"], row["vy"]

        # Determine ball team affiliation
        ball_team = 1 if (row["home_has_possession"] and row["team"] == "home") or (
            not row["home_has_possession"] and row["team"] == "away") else 0

        # Node features
        node_features.append([x, y, vx, vy, ball_team])
        node_team.append(row["team"])

        # Map node index to a unique ID
        node_id_map[idx] = len(node_id_map)

    return node_features, node_team, node_id_map

def process_edges(node_features, node_ids, fully_connected):
    """Process edges and return edge index and attributes."""
    edge_index = []
    edge_attrs = []

    for i in node_ids:
        for j in node_ids:
            if i != j:
                if fully_connected or node_features[i][4] == node_features[j][4]:  # Same team
                    xi, yi = node_features[i][0], node_features[i][1]
                    xj, yj = node_features[j][0], node_features[j][1]
                    distance = calculate_distance(xi, yi, xj, yj)
                    edge_index.append([i, j])
                    edge_attrs.append([distance])

    return edge_index, edge_attrs