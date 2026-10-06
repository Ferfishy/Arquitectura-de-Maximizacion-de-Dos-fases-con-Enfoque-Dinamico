import numpy as np
import networkx as nx

# tqdm es opcional
try:
    from tqdm import tqdm as _tqdm
except ImportError:
    def _tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        if desc:
            print(f"  [{desc}]...")
        return iterable

# ─────────────────────────────────────────────────────────────────────────────
# 1. Un paso de tiempo del modelo SIS (Vectorizado)
# ─────────────────────────────────────────────────────────────────────────────

def sis_model_spread_step_vectorized(X: np.ndarray,
                                     A_T,
                                     beta: float,
                                     delta: float) -> np.ndarray:
    """
    Ejecuta UN paso de tiempo del modelo SIS sobre el grafo usando operaciones vectorizadas.
    
    X: vector booleano (N,) donde 1=Infectado, 0=Susceptible.
    A_T: matriz de adyacencia dispersa transpuesta (scipy.sparse.csr_matrix).
    beta: probabilidad de infección (S->I).
    delta: probabilidad de recuperación (I->S).
    """
    # 1. Recuperación I -> S
    recover_mask = (X == 1) & (np.random.rand(len(X)) < delta)
    
    # 2. Propagación S -> I
    # k_inf contiene cuántos vecinos infecciosos apuntan a cada nodo
    k_inf = A_T.dot(X)
    p_inf = 1.0 - (1.0 - beta) ** k_inf
    infect_mask = (X == 0) & (np.random.rand(len(X)) < p_inf)
    
    # 3. Actualizar estado
    X_new = X.copy()
    X_new[recover_mask] = 0
    X_new[infect_mask] = 1
    
    return X_new

# ─────────────────────────────────────────────────────────────────────────────
# 2. Ganancia marginal dinámica Δ(u, t) (Vectorizado)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_dynamic_marginal_gain_vectorized(u_idx: int,
                                              X_current: np.ndarray,
                                              A_T,
                                              remaining_time: int,
                                              mc_simulations: int,
                                              beta: float,
                                              delta: float) -> float:
    ganancia_total = 0.0
    
    for _ in range(mc_simulations):
        # Trayectoria SIN u
        X_sin_u = X_current.copy()
        spread_sin_u = 0
        
        # Trayectoria CON u
        X_con_u = X_current.copy()
        X_con_u[u_idx] = 1
        spread_con_u = 0
        
        for _ in range(remaining_time):
            X_sin_u = sis_model_spread_step_vectorized(X_sin_u, A_T, beta, delta)
            spread_sin_u += X_sin_u.sum()
            
            X_con_u = sis_model_spread_step_vectorized(X_con_u, A_T, beta, delta)
            spread_con_u += X_con_u.sum()
            
        ganancia_total += (spread_con_u - spread_sin_u)
        
    return ganancia_total / mc_simulations

# ─────────────────────────────────────────────────────────────────────────────
# 3. Influencia sostenida (Vectorizado)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sustained_influence(G,
                                seeds: list,
                                T: int = 20,
                                beta: float = 0.02,
                                delta: float = 0.10,
                                mc_simulations: int = 20) -> float:
    if not seeds:
        return 0.0
        
    nodos_lista = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodos_lista)}
    N = len(nodos_lista)
    
    # Matriz A transpuesta
    A = nx.adjacency_matrix(G, nodelist=nodos_lista, weight=None)
    A_T = A.T.tocsr()
    
    total_activos = 0.0
    
    for _ in range(mc_simulations):
        X = np.zeros(N, dtype=np.int8)
        for s in seeds:
            X[node_to_idx[s]] = 1
            
        suma_pasos = 0
        for _ in range(T):
            X = sis_model_spread_step_vectorized(X, A_T, beta, delta)
            suma_pasos += X.sum()
            
        total_activos += suma_pasos / T
        
    return total_activos / mc_simulations

# ─────────────────────────────────────────────────────────────────────────────
# 4. Fase 2 — Selección dinámica bajo restricciones (Vectorizado)
# ─────────────────────────────────────────────────────────────────────────────

def phase2_sis_dynamic(G,
                       candidates: list,
                       total_budget: float,
                       time_limit: int,
                       beta: float = 0.02,
                       delta: float = 0.10,
                       mc_simulations: int = 10,
                       verbose: bool = True) -> tuple:
    """
    Fase 2: Selección Dinámica (SIS) HIPER-OPTIMIZADA (Vectorizada).
    """
    if verbose:
        print(f"\n{'='*55}")
        print(f"FASE 2: Selección Dinámica (SIS) [VECTORIZADO]")
        print(f"  Candidatos: {len(candidates):,}  |  "
              f"Presupuesto B={total_budget}  |  Deadline D={time_limit}")
        print(f"  β={beta}  |  δ={delta}  |  MC={mc_simulations}")
        print(f"{'='*55}")

    nodos_lista = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodos_lista)}
    N = len(nodos_lista)
    
    # Preparar matriz dispersa para cálculo en O(E) ultrarrápido
    A = nx.adjacency_matrix(G, nodelist=nodos_lista, weight=None)
    A_T = A.T.tocsr()
    
    presupuesto_actual = total_budget
    X_current = np.zeros(N, dtype=np.int8)
    selected_seeds = []
    sis_evals = 0

    for t in range(time_limit):
        if verbose:
            print(f"\n  [t={t}] Presupuesto restante: {presupuesto_actual:.1f}  |  "
                  f"Semillas: {len(selected_seeds)}")

        semillas_set = set(selected_seeds)
        candidatos_asequibles = [
            v for v in candidates
            if G.nodes[v]['cost'] <= presupuesto_actual
            and v not in semillas_set
        ]

        if not candidatos_asequibles:
            if verbose:
                print("  Sin candidatos asequibles o presupuesto agotado.")
            break

        if verbose:
            print(f"  Evaluando {len(candidatos_asequibles)} candidatos...")

        mejor_nodo = None
        mejor_ratio = -np.inf

        for u in candidatos_asequibles:
            costo = G.nodes[u]['cost']
            tiempo_restante = time_limit - t

            ganancia = estimate_dynamic_marginal_gain_vectorized(
                node_to_idx[u], X_current, A_T, tiempo_restante,
                mc_simulations=mc_simulations,
                beta=beta, delta=delta
            )
            sis_evals += 1

            ratio = ganancia / costo
            if ratio > mejor_ratio:
                mejor_ratio = ratio
                mejor_nodo = u

        if mejor_nodo is not None and mejor_ratio > 0:
            costo_nodo = G.nodes[mejor_nodo]['cost']
            if verbose:
                print(f"  → Seleccionado: nodo {mejor_nodo}  "
                      f"(costo={costo_nodo}, ratio={mejor_ratio:.4f})")

            selected_seeds.append(mejor_nodo)
            X_current[node_to_idx[mejor_nodo]] = 1
            presupuesto_actual -= costo_nodo

            X_current = sis_model_spread_step_vectorized(
                X_current, A_T, beta, delta
            )
            sis_evals += 1

            if verbose:
                print(f"  Nodos activos tras paso SIS: {X_current.sum()}")
        else:
            if verbose:
                print("  Ningún candidato ofreció ratio > 0. Deteniendo.")
            break

    if verbose:
        print(f"\n[Fase 2] Completada.")
        print(f"  Semillas seleccionadas ({len(selected_seeds)}): {selected_seeds}")
        costo_total = sum(G.nodes[u]['cost'] for u in selected_seeds)
        print(f"  Costo total: {costo_total:.1f} / {total_budget}")
        print(f"  Evaluaciones SIS: {sis_evals}")

    assert len(selected_seeds) == len(set(selected_seeds)), "ERROR C1: duplicados."
    return selected_seeds, sis_evals

if __name__ == '__main__':
    import os
    from graph_utils import prepare_graph, generate_dummy_graph

    DATASET = 'datasets/facebook_combined.txt'
    if os.path.exists(DATASET):
        G = prepare_graph(DATASET, seed=42)
        np.random.seed(42)
        candidatos = list(np.random.choice(list(G.nodes()), 30, replace=False))
    else:
        print("Dataset no encontrado. Usando grafo dummy.")
        G = generate_dummy_graph(n=100, seed=42)
        candidatos = list(G.nodes())[:20]

    semillas, evals = phase2_sis_dynamic(
        G, candidatos,
        total_budget=50,
        time_limit=3,
        beta=0.02,
        delta=0.10,
        mc_simulations=5,
        verbose=True
    )
