import networkx as nx
import numpy as np

def load_graph_from_edgelist(filepath, directed=True):
    """
    Carga un grafo desde un archivo de lista de aristas (formato: u v).
    """
    if directed:
        G = nx.read_edgelist(filepath, create_using=nx.DiGraph(), nodetype=int)
    else:
        G = nx.read_edgelist(filepath, create_using=nx.Graph(), nodetype=int)
        G = G.to_directed() # Convertimos a dirigido para el modelo de propagación
    return G

def assign_edge_probabilities(G, default_prob=0.1):
    """
    Asigna probabilidades de influencia a las aristas.
    Basado en el modelo de cascada independiente (IC).
    Se puede ajustar para usar la inversa del grado de entrada, por ejemplo.
    """
    for u, v in G.edges():
        # Asignamos una probabilidad p_uv
        # Alternativa común: 1 / in_degree(v)
        in_degree = G.in_degree(v)
        prob = 1.0 / in_degree if in_degree > 0 else default_prob
        G[u][v]['weight'] = prob
    return G

def assign_node_costs(G, cost_factor=1.0):
    """
    Asigna un costo de activación a cada nodo proporcional a su grado total.
    """
    for n in G.nodes():
        degree = G.degree(n)
        # Aseguramos un costo mínimo de 1 para evitar divisiones por cero o nodos gratis
        cost = max(1, int(degree * cost_factor))
        G.nodes[n]['cost'] = cost
    return G

def prepare_graph(filepath):
    """
    Pipeline de preparación del grafo.
    """
    print(f"Cargando grafo desde {filepath}...")
    G = load_graph_from_edgelist(filepath)
    print(f"Nodos: {G.number_of_nodes()}, Aristas: {G.number_of_edges()}")
    
    G = assign_edge_probabilities(G)
    G = assign_node_costs(G, cost_factor=1.5) # Factor ajustable
    
    return G

# Generador de grafo dummy para pruebas mientras no haya dataset real
def generate_dummy_graph():
    print("Generando grafo aleatorio (Barabási-Albert) para prueba piloto...")
    G_undirected = nx.barabasi_albert_graph(100, 3)
    G = G_undirected.to_directed()
    G = assign_edge_probabilities(G)
    G = assign_node_costs(G, cost_factor=1.0)
    return G
