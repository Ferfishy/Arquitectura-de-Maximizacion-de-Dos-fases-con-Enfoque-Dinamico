"""
graph_utils.py
==============
Utilidades de carga, preprocesamiento y muestreo del grafo para el sistema
de Maximización de Influencia en Dos Fases.

Funciones principales:
  - load_graph_from_edgelist : carga desde archivo de lista de aristas.
  - assign_ic_weights        : pesos Weighted Cascade para el modelo IC.
  - assign_node_costs        : costo proporcional al grado, normalizado.
  - sample_connected_subgraph: extrae un subgrafo conexo de n nodos (BFS).
  - prepare_graph            : pipeline completo de preparación.
  - generate_dummy_graph     : grafo Barabási-Albert para pruebas rápidas.
"""

import networkx as nx
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1. Carga del grafo
# ─────────────────────────────────────────────────────────────────────────────

def load_graph_from_edgelist(filepath: str) -> nx.DiGraph:
    """
    Carga un grafo no dirigido desde un archivo de lista de aristas en formato
    SNAP (líneas con comentarios '#' y pares 'u v') y lo convierte a dirigido.

    El dataset de Facebook SNAP está en formato no dirigido; al convertirlo
    a DiGraph cada arista (u,v) genera tanto (u→v) como (v→u).

    Parameters
    ----------
    filepath : str
        Ruta al archivo de aristas (p. ej. 'facebook_combined.txt').

    Returns
    -------
    nx.DiGraph
        Grafo dirigido con nodos enteros.
    """
    # nx.read_edgelist ignora las líneas que empiezan con '#' por defecto.
    G_und = nx.read_edgelist(filepath, create_using=nx.Graph(), nodetype=int,
                              comments='#')
    G = G_und.to_directed()
    return G


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pesos de aristas — Modelo IC (Weighted Cascade)
# ─────────────────────────────────────────────────────────────────────────────

def assign_ic_weights(G: nx.DiGraph, default_prob: float = 0.1) -> nx.DiGraph:
    """
    Asigna probabilidades de influencia IC usando el modelo Weighted Cascade:
        p(u→v) = 1 / in_degree(v)

    Estos pesos se usan EXCLUSIVAMENTE en la Fase 1 (IC / CELF).
    La Fase 2 (SIS) usa sus propios parámetros β y δ globales,
    sin acceder a estos pesos.

    Parameters
    ----------
    G            : DiGraph sobre el que se escriben los atributos.
    default_prob : prob. de respaldo si in_degree(v) == 0.

    Returns
    -------
    G con atributo 'weight' en cada arista.
    """
    for u, v in G.edges():
        in_deg = G.in_degree(v)
        G[u][v]['weight'] = 1.0 / in_deg if in_deg > 0 else default_prob
    return G


# Alias mantenido por compatibilidad con código antiguo
def assign_edge_probabilities(G: nx.DiGraph,
                              default_prob: float = 0.1) -> nx.DiGraph:
    """Alias de assign_ic_weights (compatibilidad hacia atrás)."""
    return assign_ic_weights(G, default_prob)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Costos de nodos — proporcional al grado normalizado
# ─────────────────────────────────────────────────────────────────────────────

def assign_node_costs(G: nx.DiGraph,
                      escala: float = 12.0) -> nx.DiGraph:
    """
    Asigna un costo de activación a cada nodo según la fórmula:

        c(u) = 1 + escala * (grado(u) / grado_max)

    redondeado al entero más próximo, con mínimo 1.
    'grado' es el grado total (entrante + saliente en el DiGraph).

    Con escala=12 el nodo de mayor grado cuesta 13 y el mínimo cuesta 1.

    Parameters
    ----------
    G      : DiGraph al que se escriben los atributos.
    escala : amplitud del rango de costos (por defecto 12).

    Returns
    -------
    G con atributo 'cost' en cada nodo (entero ≥ 1).
    """
    grados = dict(G.degree())                     # grado total de cada nodo
    grado_max = max(grados.values()) if grados else 1

    for n, deg in grados.items():
        costo = 1.0 + escala * (deg / grado_max)
        G.nodes[n]['cost'] = max(1, round(costo))

    return G


# ─────────────────────────────────────────────────────────────────────────────
# 4. Muestreo de subgrafos conexos (para experimentos de escalabilidad)
# ─────────────────────────────────────────────────────────────────────────────

def sample_connected_subgraph(G: nx.DiGraph,
                               n: int,
                               seed: int = 42) -> nx.DiGraph:
    """
    Extrae un subgrafo conexo de exactamente n nodos mediante BFS desde
    el nodo de mayor grado (hub central), garantizando conexidad.

    Si n >= |V|, devuelve el grafo completo.

    Parameters
    ----------
    G    : DiGraph fuente (ya preprocesado con pesos y costos).
    n    : número de nodos del subgrafo deseado.
    seed : semilla aleatoria (actualmente no usada, BFS es determinista).

    Returns
    -------
    nx.DiGraph subgrafo inducido de n nodos con atributos preservados.
    """
    if n >= G.number_of_nodes():
        return G.copy()

    # Nodo raíz: el de mayor grado (más conectado → BFS cubre más nodos)
    nodo_raiz = max(G.degree(), key=lambda x: x[1])[0]

    # BFS sobre el grafo subyacente no dirigido para garantizar conexidad
    G_und = G.to_undirected()
    nodos_bfs = []
    visitados = {nodo_raiz}
    cola = [nodo_raiz]

    while cola and len(nodos_bfs) < n:
        u = cola.pop(0)
        nodos_bfs.append(u)
        # Ordenar vecinos por grado descendente para favorecer hubs
        vecinos = sorted(G_und.neighbors(u),
                         key=lambda v: G_und.degree(v), reverse=True)
        for v in vecinos:
            if v not in visitados:
                visitados.add(v)
                cola.append(v)

    subgrafo = G.subgraph(nodos_bfs[:n]).copy()
    return subgrafo


# ─────────────────────────────────────────────────────────────────────────────
# 5. Pipeline de preparación completa
# ─────────────────────────────────────────────────────────────────────────────

def prepare_graph(filepath: str,
                  escala_costos: float = 12.0,
                  seed: int = 42) -> nx.DiGraph:
    """
    Pipeline completo:
      1. Carga el archivo de aristas como DiGraph.
      2. Asigna pesos IC (Weighted Cascade).
      3. Asigna costos de nodo normalizados.

    Parameters
    ----------
    filepath      : ruta al dataset de aristas.
    escala_costos : parámetro de escala para la fórmula de costo (default 12).
    seed          : semilla aleatoria global (se aplica a numpy).

    Returns
    -------
    nx.DiGraph completamente preparado.
    """
    np.random.seed(seed)

    print(f"[graph_utils] Cargando grafo desde '{filepath}'...")
    G = load_graph_from_edgelist(filepath)
    print(f"[graph_utils] Nodos: {G.number_of_nodes():,}  "
          f"Aristas: {G.number_of_edges():,}")

    G = assign_ic_weights(G)
    G = assign_node_costs(G, escala=escala_costos)

    # Estadísticas de costos para verificación
    costos = [G.nodes[n]['cost'] for n in G.nodes()]
    print(f"[graph_utils] Costo mínimo: {min(costos)}  "
          f"Costo máximo: {max(costos)}  "
          f"Costo promedio: {np.mean(costos):.2f}")

    return G


# ─────────────────────────────────────────────────────────────────────────────
# 6. Generador de grafo dummy (pruebas sin dataset)
# ─────────────────────────────────────────────────────────────────────────────

def generate_dummy_graph(n: int = 100,
                         m: int = 3,
                         seed: int = 42) -> nx.DiGraph:
    """
    Genera un grafo Barabási-Albert dirigido para pruebas rápidas.
    Útil cuando el dataset real no está disponible.

    Parameters
    ----------
    n    : número de nodos.
    m    : aristas por nodo nuevo (parámetro BA).
    seed : semilla aleatoria.
    """
    print(f"[graph_utils] Generando grafo BA de prueba "
          f"({n} nodos, m={m})...")
    np.random.seed(seed)
    G_und = nx.barabasi_albert_graph(n, m, seed=seed)
    G = G_und.to_directed()
    G = assign_ic_weights(G)
    G = assign_node_costs(G)
    return G


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución directa — verificación rápida
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os

    DATASET = 'facebook_combined.txt'
    if os.path.exists(DATASET):
        G = prepare_graph(DATASET, seed=42)
        # Verificar algunos atributos
        nodo_ejemplo = list(G.nodes())[0]
        arista_ejemplo = list(G.edges())[0]
        print(f"\n[Verificación] Nodo {nodo_ejemplo}: "
              f"cost={G.nodes[nodo_ejemplo]['cost']}")
        print(f"[Verificación] Arista {arista_ejemplo}: "
              f"weight={G[arista_ejemplo[0]][arista_ejemplo[1]]['weight']:.4f}")

        # Subgrafos de muestra
        for tam in [200, 500, 1000]:
            sg = sample_connected_subgraph(G, tam, seed=42)
            print(f"[Subgrafo {tam}] Nodos reales: {sg.number_of_nodes()}  "
                  f"Aristas: {sg.number_of_edges()}")
    else:
        print(f"Dataset '{DATASET}' no encontrado. Usando grafo dummy.")
        G = generate_dummy_graph(seed=42)
        print(f"Grafo dummy listo: {G.number_of_nodes()} nodos, "
              f"{G.number_of_edges()} aristas.")
