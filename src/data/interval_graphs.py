import torch
from torch_geometric.data import Data

def process_interval(args, fully_connected=False):
    """
    Process a single interval and transform it into a PyTorch Geometric graph.

    Args:
        args: Tuple containing (interval_id, interval_df).

    Returns:
        PyTorch Geometric Data object.
    """
    interval_id, interval_df = args

    # Step 1: Initialize graph components
    node_features = []  # List to hold node features
    edge_index = []     # List to hold edge connections
    edge_attrs = []     # List to hold edge attributes
    node_team = []      # Track the team of each node for edge attributes

    # Step 2: Process nodes
    node_id_map = {}  # Map to assign a unique ID to each node
    for idx, row in interval_df.iterrows():
        # Example node features: x, y, vx, vy (customize as needed)
        x, y = row["x"], row["y"]
        vx, vy = row["vx"], row["vy"]

        ball_team = (row["home_has_possession"] and row["team"]=="home")   # Team affiliation of the ball

        # TODO: polar coordinates to ball or to goal

        node_features.append([x, y, vx, vy, ball_team])  # Node features
        node_team.append(row["team"])  # Team affiliation

        # Assign a unique ID to each node
        node_id_map[idx] = len(node_id_map)

    # Step 3: Process edges (fully connected graph for simplicity)
    node_ids = list(node_id_map.values())
    for i in node_ids:
        for j in node_ids:
            if i != j:
                if fully_conected or (node_features[i][4] and node_features[j][4]):  # Connect if same team
                    edge_index.append([i, j])

                    # Edge attributes: Eucledian distance between nodes
                    xi, yi = node_features[i][0], node_features[i][1]
                    xj, yj = node_features[j][0], node_features[j][1]
                    distance = ((xi - xj) ** 2 + (yi - yj) ** 2) ** 0.5
                    edge_attrs.append([distance])

    # Step 4: Convert lists to PyTorch tensors
    node_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)

    # Step 5: Create a PyTorch Geometric Data object
    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attrs)
    data.interval_id = interval_id  # Store the interval ID for reference

    return data
