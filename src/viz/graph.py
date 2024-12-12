import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

def plot_graph(data):
    """
    Plots the graph with nodes separated by team

    Args:
        data: PyTorch Geometric Data object containing graph information.
    """
    # Convert to NetworkX graph
    G = to_networkx(data, to_undirected=True)

    positions = data.x[:, :2].numpy()  # First two features are x and y coordinates

    # Create a dictionary for node positions
    pos = {i: (positions[i, 0], positions[i, 1]) for i in range(len(positions))}

    # Extract node features (team affiliation is the 4th feature: x[:, 3])
    team_affiliation = data.x[:, 4].numpy()  # Convert to numpy for easy handling

    # Separate nodes by team
    team_1_nodes = [i for i, team in enumerate(team_affiliation) if team == 1]
    team_0_nodes = [i for i, team in enumerate(team_affiliation) if team == 0]

    # Define node colors: Red for team 1, Blue for team 0
    color_map = []
    for team in team_affiliation:
        color_map.append('red' if team == 1 else 'blue')

    # Layout to separate teams
    #pos = nx.spring_layout(G, seed=42)  # You can try other layouts like bipartite_layout

    # Plot the graph
    plt.figure(figsize=(10, 10))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=300,
        node_color=color_map,
        edge_color="gray"
    )
    plt.title(f"Interval ID: {data.interval_id}")
    plt.show()

