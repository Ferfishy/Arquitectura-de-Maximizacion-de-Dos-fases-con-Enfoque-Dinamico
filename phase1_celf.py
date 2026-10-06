"""
phase1_celf.py
==============
Fase 1 del sistema de Maximización de Influencia en Dos Fases.

Implementa el filtro voraz estático con el modelo Independent Cascade (IC)
usando la estructura CELF (Cost-Effective Lazy Forward) para reducir el grafo
a un conjunto de k' nodos candidatos.

Dos modos de operación:
  - fast_mode=True  (default, para grafos grandes como Facebook):
      Un solo barrido de ganancias individuales Δ(u|∅); retorna top-k'.
      Complejidad O(|V| · mc) en lugar de O(k'·|V|·mc).
  - fast_mode=False (para redes pequeñas):
      CELF completo con re-evaluación de ganancia marginal, aprovechando
      la submodularidad para reducir simulaciones MC.

INVARIANTE: la difusión IC siempre corre sobre el grafo COMPLETO G;
nunca se restringe la red de propagación.
"""

import heapq
import numpy as np

# tqdm es opcional (puede no estar instalado en todos los entornos)
try:
    from tqdm import tqdm as _tqdm
    _TQDM_DISPONIBLE = True
except ImportError:
    _TQDM_DISPONIBLE = False

    def _tqdm(iterable, **kwargs):
        """Stub de tqdm que itera sin barra de progreso."""
        desc = kwargs.get('desc', '')
        if desc:
            print(f"  [{desc}] procesando...")
        return iterable


# ─────────────────────────────────────────────────────────────────────────────
# 1. Difusión IC — Monte Carlo sobre el grafo completo
# ─────────────────────────────────────────────────────────────────────────────

def ic_model_spread(G, seeds, mc_simulations: int = 100) -> float:
    """
    Estima el spread esperado de un conjunto de semillas bajo el modelo IC.

    Cada simulación activa las semillas y propaga en cascada; un nodo v se
    activa desde u con probabilidad G[u][v]['weight'] (Weighted Cascade).
    Retorna el promedio de nodos activos al finalizar la cascada.

    INVARIANTE: la simulación corre siempre sobre G completo; el conjunto
    'seeds' sólo define los nodos inicialmente activos.

    Parameters
    ----------
    G              : nx.DiGraph con atributo 'weight' en aristas.
    seeds          : lista/set de nodos semilla.
    mc_simulations : número de simulaciones Monte Carlo.

    Returns
    -------
    float : spread promedio (nodos influenciados, incluidas las semillas).
    """
    total_spread = 0

    for _ in range(mc_simulations):
        activos = set(seeds)
        nuevos_activos = set(seeds)

        while nuevos_activos:
            siguiente_ronda = set()
            for u in nuevos_activos:
                for v in G.successors(u):
                    if v not in activos:
                        # Usa el peso IC (Weighted Cascade) de la arista
                        prob = G[u][v].get('weight', 0.1)
                        if np.random.rand() < prob:
                            siguiente_ronda.add(v)
            activos.update(siguiente_ronda)
            nuevos_activos = siguiente_ronda

        total_spread += len(activos)

    return total_spread / mc_simulations


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fase 1 — Modo rápido (un solo barrido de ganancias individuales)
# ─────────────────────────────────────────────────────────────────────────────

def _filtro_rapido(G, k_prime: int, mc_simulations: int) -> list:
    """
    Selecciona top-k' candidatos evaluando la ganancia individual Δ(u|∅)
    para cada nodo con mc_simulations Monte Carlo.

    Complejidad: O(|V| · mc_simulations · |E|/|V|) — lineal en nodos.
    Adecuado para grafos con miles de nodos donde el CELF iterativo es lento.

    Returns
    -------
    list : k' nodos ordenados por spread individual descendente.
    """
    print(f"  [fast_mode] Calculando spread individual para "
          f"{G.number_of_nodes():,} nodos "
          f"({mc_simulations} MC por nodo)...")

    ganancias = {}  # nodo → spread individual

    nodos = list(G.nodes())
    for u in _tqdm(nodos, desc="Barrido IC individual"):
        ganancias[u] = ic_model_spread(G, [u], mc_simulations)

    # Ordenar y retornar los k' mejores
    candidatos = sorted(ganancias, key=lambda x: ganancias[x], reverse=True)
    candidatos = candidatos[:k_prime]

    spreads_top = [ganancias[u] for u in candidatos[:5]]
    print(f"  [fast_mode] Top-5 spreads individuales: "
          f"{[f'{s:.1f}' for s in spreads_top]}")

    return candidatos


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fase 1 — CELF completo (para redes pequeñas)
# ─────────────────────────────────────────────────────────────────────────────

def _celf_completo(G, k_prime: int, mc_simulations: int) -> list:
    """
    Algoritmo CELF (Leskovec et al., 2007) completo.

    Aprovecha la submodularidad del spread IC: si la ganancia marginal de u
    calculada en iteración anterior sigue siendo mayor que el mejor spread en
    la cola, u puede seleccionarse sin re-evaluación.

    Adecuado para grafos con cientos de nodos.

    Returns
    -------
    list : hasta k' nodos en orden de selección greedy.
    """
    print(f"  [CELF] Precalculando ganancias individuales para "
          f"{G.number_of_nodes()} nodos...")

    # Paso 1: evaluar ganancia individual de cada nodo y construir cola
    Q = []  # min-heap con (−spread, nodo)
    for u in _tqdm(G.nodes(), desc="IC individual (CELF init)"):
        spread = ic_model_spread(G, [u], mc_simulations)
        heapq.heappush(Q, (-spread, u))

    candidatos = []
    spread_candidatos = 0.0  # spread acumulado del conjunto actual

    print(f"  [CELF] Seleccionando {k_prime} candidatos con lazy forward...")

    # Primer nodo: es el máximo global; no requiere re-evaluación
    if Q:
        neg_spread, mejor_nodo = heapq.heappop(Q)
        candidatos.append(mejor_nodo)
        spread_candidatos = -neg_spread

    # Pasos siguientes: re-evaluar sólo cuando es necesario
    iteraciones_celf = 0
    iteraciones_reeval = 0

    while len(candidatos) < k_prime and Q:
        iteraciones_celf += 1
        neg_spread_anterior, u = heapq.heappop(Q)

        # Re-calcular ganancia marginal de u dado el conjunto actual
        spread_con_u = ic_model_spread(G, candidatos + [u], mc_simulations)
        ganancia_marginal = spread_con_u - spread_candidatos
        iteraciones_reeval += 1

        # Comprobar si sigue siendo el mejor (lazy check)
        if Q:
            neg_spread_siguiente, _ = Q[0]
            if ganancia_marginal >= -neg_spread_siguiente:
                # u sigue siendo el mejor; añadir
                candidatos.append(u)
                spread_candidatos = spread_con_u
            else:
                # u ya no es el mejor con la ganancia marginal actualizada;
                # reinsertar con su ganancia marginal real
                heapq.heappush(Q, (-ganancia_marginal, u))
        else:
            # No hay más candidatos; añadir u directamente
            candidatos.append(u)
            spread_candidatos = spread_con_u

    print(f"  [CELF] Re-evaluaciones realizadas: {iteraciones_reeval} "
          f"de {iteraciones_celf} posibles.")
    return candidatos


# ─────────────────────────────────────────────────────────────────────────────
# 4. Función principal de Fase 1
# ─────────────────────────────────────────────────────────────────────────────

def phase1_celf(G,
                k_prime: int,
                mc_simulations: int = 50,
                fast_mode: bool = True) -> list:
    """
    Fase 1: Filtrado Voraz Estático.

    Reduce el conjunto de candidatos de |V| a k' nodos usando el modelo IC.
    La difusión IC se simula SIEMPRE sobre el grafo completo G.

    Parameters
    ----------
    G              : nx.DiGraph con pesos 'weight' y costos 'cost'.
    k_prime        : número de candidatos a retornar (|V'| = k').
    mc_simulations : simulaciones Monte Carlo por evaluación IC.
    fast_mode      : si True, usa un solo barrido (O(|V|)); si False, CELF
                     completo con re-evaluación de ganancia marginal.

    Returns
    -------
    list : k' nodos candidatos (sin orden particular en fast_mode).
    """
    print(f"\n{'='*55}")
    print(f"FASE 1: Filtrado Voraz Estático ({'rápido' if fast_mode else 'CELF completo'})")
    print(f"  Nodos en G: {G.number_of_nodes():,}  |  "
          f"k' objetivo: {k_prime}  |  MC: {mc_simulations}")
    print(f"{'='*55}")

    if fast_mode:
        candidatos = _filtro_rapido(G, k_prime, mc_simulations)
    else:
        candidatos = _celf_completo(G, k_prime, mc_simulations)

    # Asegurar que no retornamos más de k_prime
    candidatos = candidatos[:k_prime]

    print(f"\n[Fase 1] Completada: {len(candidatos)} candidatos extraídos "
          f"de {G.number_of_nodes():,} nodos totales.")
    return candidatos


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución directa — prueba rápida
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    from graph_utils import prepare_graph, generate_dummy_graph

    DATASET = 'datasets/facebook_combined.txt'
    if os.path.exists(DATASET):
        G = prepare_graph(DATASET, seed=42)
        # Prueba con fast_mode en el grafo completo
        candidatos = phase1_celf(G, k_prime=20, mc_simulations=20,
                                  fast_mode=True)
        print(f"\nCandidatos (primeros 10): {candidatos[:10]}")
    else:
        print("Dataset no encontrado. Usando grafo dummy (100 nodos).")
        G = generate_dummy_graph(n=100, seed=42)
        # Prueba con CELF completo en grafo pequeño
        candidatos = phase1_celf(G, k_prime=10, mc_simulations=30,
                                  fast_mode=False)
        print(f"\nCandidatos: {candidatos}")
