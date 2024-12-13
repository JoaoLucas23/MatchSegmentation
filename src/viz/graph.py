import matplotlib.pyplot as plt
import networkx as nx
import imageio
import os

def plot_graph(G, show=True):
    """
    Plota um grafo NetworkX, separando os nós por cor de acordo com o atributo 'ball_team'.

    Args:
        G (networkx.Graph): Grafo com atributos de nós 'x', 'y', 'ball_team', etc.
        show (bool): Se True, mostra o plot interativamente. Se False, apenas cria o plot (para salvar).
    """
    # Extrai a posição dos nós a partir dos atributos x, y
    pos = {n: (d['x'], d['y']) for n, d in G.nodes(data=True)}

    # Extrai o atributo ball_team para definir cor
    ball_teams = nx.get_node_attributes(G, 'ball_team')
    # Define as cores: vermelho para ball_team=1, azul para ball_team=0
    colors = ['red' if ball_teams[n] == 1 else 'blue' for n in G.nodes()]

    # Título do gráfico
    interval_id = G.graph.get('interval_id', 'unknown')

    plt.figure(figsize=(10, 10))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=300,
        node_color=colors,
        edge_color="gray"
    )
    plt.title(f"Interval ID: {interval_id}")

    if show:
        plt.show()


def plot_graph_sequence(graph_list, out_gif='sequence.gif', duration=0.5):
    """
    Plota uma sequência de grafos NetworkX (por exemplo, frames no tempo) e cria um GIF animado.

    Args:
        graph_list (list): Lista de grafos NetworkX. Cada grafo deve ter 'x', 'y' por nó e
                           idealmente 'ball_team' e 'interval_id'.
        out_gif (str): Nome do arquivo GIF de saída.
        duration (float): Duração entre frames no GIF.
    """
    filenames = []

    # Gera um frame (imagem) para cada grafo na lista
    for i, G in enumerate(graph_list):
        plt.figure(figsize=(10, 10))
        pos = {n: (d['x'], d['y']) for n, d in G.nodes(data=True)}
        ball_teams = nx.get_node_attributes(G, 'ball_team')
        colors = ['red' if ball_teams[n] == 1 else 'blue' for n in G.nodes()]

        nx.draw(G, pos, with_labels=True, node_size=300, node_color=colors, edge_color="gray")

        interval_id = G.graph.get('interval_id', i)
        plt.title(f"Interval ID: {interval_id}")

        fname = f"frame_{i}.png"
        plt.savefig(fname)
        plt.close()
        filenames.append(fname)

    # Cria o GIF a partir dos frames salvos
    with imageio.get_writer(out_gif, mode='I', duration=duration) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

    # Opcional: remover os arquivos temporários de frames
    for filename in filenames:
        os.remove(filename)
