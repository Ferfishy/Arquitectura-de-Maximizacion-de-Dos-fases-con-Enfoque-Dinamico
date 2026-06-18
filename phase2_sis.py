"""
phase2_sis.py
=============
Fase 2 del sistema de Maximización de Influencia en Dos Fases.

Implementa la selección dinámica de semillas bajo restricciones de presupuesto
B y deadline D usando el modelo SIS (Susceptible-Infectious-Susceptible).

CORRECCIONES APLICADAS
──────────────────────
C1 — Bug de re-selección (doble cobro de presupuesto):
     El filtro de candidatos asequibles ahora usa `v not in set(selected_seeds)`,
     que es la lista permanente de nodos ya comprados. Esto garantiza que un
     nodo no se pueda volver a comprar aunque el SIS lo recupere y salga de
     current_active_nodes. Formalización: u ∈ V' \\ S.

C2 — β y δ separados (modelo SIS formal):
     - β (beta)  = probabilidad de que un nodo Infeccioso active a un vecino
                   Susceptible en cada paso de tiempo (S→I). Parámetro global,
                   independiente del peso IC de las aristas.
     - δ (delta) = probabilidad de que un nodo Infeccioso se recupere (I→S).
     El peso 'weight' de las aristas sigue perteneciendo al modelo IC de la
     Fase 1 y NO se usa en ninguna función de esta Fase 2.

C3 — Capa de análisis añadida:
     (a) compute_sustained_influence : influencia media del conjunto final
         de semillas en la ventana T, simulado sobre el grafo COMPLETO.
     (b) La Fase 2 ahora retorna (selected_seeds, sis_evals) para medir
         carga computacional.

INVARIANTE: la difusión SIS siempre corre sobre el grafo COMPLETO G;
la Fase 1 sólo restringe qué nodos pueden ser elegidos como semilla.
"""

import numpy as np

# tqdm es opcional
try:
    from tqdm import tqdm as _tqdm
    _TQDM_DISPONIBLE = True
except ImportError:
    _TQDM_DISPONIBLE = False

    def _tqdm(iterable, **kwargs):
        """Stub de tqdm sin barra de progreso."""
        desc = kwargs.get('desc', '')
        if desc:
            print(f"  [{desc}]...")
        return iterable


# ─────────────────────────────────────────────────────────────────────────────
# 1. Un paso de tiempo del modelo SIS
# ─────────────────────────────────────────────────────────────────────────────

def sis_model_spread_step(G,
                          active_nodes: set,
                          beta: float = 0.02,
                          delta: float = 0.10) -> set:
    """
    Ejecuta UN paso de tiempo del modelo SIS sobre el grafo completo G.

    Transiciones:
      S → I : un nodo susceptible v se infecta con probabilidad β por
              CADA vecino infeccioso u (independiente del peso IC de (u,v)).
      I → S : un nodo infeccioso u se recupera con probabilidad δ.

    CORRECCIÓN C2: se usa β explícito en lugar de G[u][v]['weight'].
    INVARIANTE   : itera G.successors(u) para todo u ∈ V (grafo completo).

    Parameters
    ----------
    G            : nx.DiGraph — grafo completo de propagación.
    active_nodes : set de nodos en estado Infeccioso (I) al inicio del paso.
    beta         : probabilidad de infección por arista (S→I), global.
    delta        : probabilidad de recuperación por nodo (I→S), global.

    Returns
    -------
    set : nuevo conjunto de nodos en estado I al final del paso.
    """
    activos_actuales = set(active_nodes)
    nuevos_activos = set(active_nodes)   # comenzamos con los ya infectados

    # ── Propagación S → I ──────────────────────────────────────────────────
    for u in activos_actuales:
        for v in G.successors(u):
            if v not in activos_actuales:
                # CORRECCIÓN C2: β global, NO G[u][v]['weight']
                if np.random.rand() < beta:
                    nuevos_activos.add(v)

    # ── Recuperación I → S ─────────────────────────────────────────────────
    recuperados = set()
    for u in activos_actuales:
        if np.random.rand() < delta:
            recuperados.add(u)

    nuevos_activos.difference_update(recuperados)
    return nuevos_activos


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ganancia marginal dinámica Δ(u, t)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_dynamic_marginal_gain(G,
                                   u,
                                   current_active_nodes: set,
                                   remaining_time: int,
                                   mc_simulations: int = 10,
                                   beta: float = 0.02,
                                   delta: float = 0.10) -> float:
    """
    Estima la ganancia marginal dinámica de activar el nodo u en el instante t:

        Δ(u, t) = E[Σ_{τ=t}^{D} |I_{τ}^{S∪{u}}|] − E[Σ_{τ=t}^{D} |I_{τ}^{S}|]

    Se proyectan 'remaining_time' pasos hacia adelante con mc_simulations
    realizaciones del SIS sobre el grafo completo G.

    CORRECCIÓN C2: delega en sis_model_spread_step con β y δ explícitos.

    Parameters
    ----------
    G                   : nx.DiGraph — grafo completo.
    u                   : nodo candidato a activar.
    current_active_nodes: conjunto actual de nodos infecciosos.
    remaining_time      : número de pasos restantes hasta D.
    mc_simulations      : realizaciones Monte Carlo.
    beta                : parámetro β del SIS.
    delta               : parámetro δ del SIS.

    Returns
    -------
    float : ganancia marginal esperada acumulada en [t, D].
    """
    ganancia_total = 0.0

    for _ in range(mc_simulations):
        # Trayectoria SIN u
        activos_sin_u = set(current_active_nodes)
        spread_sin_u = 0

        # Trayectoria CON u (añadir u como nuevo infeccioso)
        activos_con_u = set(current_active_nodes)
        activos_con_u.add(u)
        spread_con_u = 0

        for _ in range(remaining_time):
            activos_sin_u = sis_model_spread_step(G, activos_sin_u, beta, delta)
            spread_sin_u += len(activos_sin_u)

            activos_con_u = sis_model_spread_step(G, activos_con_u, beta, delta)
            spread_con_u += len(activos_con_u)

        ganancia_total += (spread_con_u - spread_sin_u)

    return ganancia_total / mc_simulations


# ─────────────────────────────────────────────────────────────────────────────
# 3. Influencia sostenida (Corrección C3a)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sustained_influence(G,
                                seeds: list,
                                T: int = 20,
                                beta: float = 0.02,
                                delta: float = 0.10,
                                mc_simulations: int = 20) -> float:
    """
    Calcula la influencia sostenida del conjunto de semillas sobre el grafo
    COMPLETO G: promedio de nodos activos a lo largo de la ventana [1, T].

        IS(S) = (1/T) · E[Σ_{t=1}^{T} |I_t^S|]

    Esto mide la calidad real del conjunto S elegido como semilla.
    Se simula mc_simulations veces para reducir varianza.

    INVARIANTE: G completo — no se restringe la red de propagación.

    Parameters
    ----------
    G              : nx.DiGraph completo.
    seeds          : lista de nodos semilla (conjunto final S).
    T              : horizonte de tiempo.
    beta           : parámetro β del SIS.
    delta          : parámetro δ del SIS.
    mc_simulations : realizaciones Monte Carlo.

    Returns
    -------
    float : influencia sostenida promedio IS(S).
    """
    if not seeds:
        return 0.0

    total_activos = 0.0

    for _ in range(mc_simulations):
        activos = set(seeds)
        suma_pasos = 0

        for _ in range(T):
            activos = sis_model_spread_step(G, activos, beta, delta)
            suma_pasos += len(activos)

        total_activos += suma_pasos / T   # promedio en ventana T

    return total_activos / mc_simulations


# ─────────────────────────────────────────────────────────────────────────────
# 4. Fase 2 — Selección dinámica bajo restricciones
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
    Fase 2: Selección Dinámica Adaptativa bajo Restricciones.

    En cada instante t ∈ {0, …, D-1}:
      1. Filtra candidatos asequibles: costo ≤ presupuesto Y no seleccionados.
      2. Evalúa la razón beneficio-costo Δ(u,t)/c(u,t) para cada candidato.
      3. Selecciona el nodo de mayor razón (si razón > 0).
      4. Avanza el estado SIS un paso para reflejar la dinámica real.

    CORRECCIÓN C1: el filtro usa `v not in set(selected_seeds)` en vez de
    `v not in current_active_nodes`, evitando la re-selección y el doble cobro.

    CORRECCIÓN C2: delega en sis_model_spread_step con β y δ explícitos.

    Parameters
    ----------
    G              : nx.DiGraph completo (propagación y atributos 'cost').
    candidates     : lista de nodos candidatos (V' ⊆ V, output de Fase 1;
                     o list(G.nodes()) para el baseline sin filtro).
    total_budget   : presupuesto B disponible.
    time_limit     : deadline D (número máximo de rondas de selección).
    beta           : parámetro β del SIS (activación).
    delta          : parámetro δ del SIS (recuperación).
    mc_simulations : realizaciones MC por evaluación Δ(u,t).
    verbose        : si True, imprime progreso por ronda.

    Returns
    -------
    tuple(list, int):
      - selected_seeds : lista de nodos seleccionados (únicos, sin duplicados).
      - sis_evals      : número total de evaluaciones SIS realizadas
                         (métrica de carga computacional).
    """
    if verbose:
        print(f"\n{'='*55}")
        print(f"FASE 2: Selección Dinámica (SIS)")
        print(f"  Candidatos: {len(candidates):,}  |  "
              f"Presupuesto B={total_budget}  |  Deadline D={time_limit}")
        print(f"  β={beta}  |  δ={delta}  |  MC={mc_simulations}")
        print(f"{'='*55}")

    presupuesto_actual = total_budget
    current_active_nodes = set()   # estado dinámico de la red (puede cambiar)
    selected_seeds = []             # conjunto S: nodos ya comprados (permanente)
    sis_evals = 0                   # contador de evaluaciones SIS

    # Conjunto inmutable de semillas para el filtro C1
    # (se convierte a set en cada ronda para búsqueda O(1))
    for t in range(time_limit):
        if verbose:
            print(f"\n  [t={t}] Presupuesto restante: {presupuesto_actual:.1f}  |  "
                  f"Semillas: {len(selected_seeds)}")

        # ── CORRECCIÓN C1: excluir nodos YA SELECCIONADOS (no sólo activos) ─
        # Formalmente: candidatos asequibles = {v ∈ V' | c(v) ≤ B_t, v ∉ S}
        semillas_set = set(selected_seeds)
        candidatos_asequibles = [
            v for v in candidates
            if G.nodes[v]['cost'] <= presupuesto_actual
            and v not in semillas_set          # ← C1: v ∉ S (no v ∉ activos)
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

            # Ganancia marginal Δ(u, t) proyectada al horizonte D − t
            # CORRECCIÓN C2: usa β y δ explícitos
            ganancia = estimate_dynamic_marginal_gain(
                G, u, current_active_nodes, tiempo_restante,
                mc_simulations=mc_simulations,
                beta=beta,
                delta=delta
            )
            sis_evals += 1

            # Razón beneficio-costo Δ(u,t)/c(u,t)
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
            current_active_nodes.add(mejor_nodo)
            presupuesto_actual -= costo_nodo

            # Avanzar el estado SIS un paso (dinámica real de la red)
            # CORRECCIÓN C2: β y δ explícitos
            current_active_nodes = sis_model_spread_step(
                G, current_active_nodes, beta, delta
            )
            sis_evals += 1   # contamos el paso de avance también

            if verbose:
                print(f"  Nodos activos tras paso SIS: "
                      f"{len(current_active_nodes)}")
        else:
            if verbose:
                print("  Ningún candidato ofreció ratio > 0. Deteniendo.")
            break

    if verbose:
        print(f"\n[Fase 2] Completada.")
        print(f"  Semillas seleccionadas ({len(selected_seeds)}): "
              f"{selected_seeds}")
        costo_total = sum(G.nodes[u]['cost'] for u in selected_seeds)
        print(f"  Costo total: {costo_total:.1f} / {total_budget}")
        print(f"  Evaluaciones SIS: {sis_evals}")

    # Verificar invariante C1: sin duplicados
    assert len(selected_seeds) == len(set(selected_seeds)), \
        "ERROR C1: selected_seeds contiene nodos duplicados."

    return selected_seeds, sis_evals


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución directa — prueba rápida
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    from graph_utils import prepare_graph, generate_dummy_graph

    DATASET = 'facebook_combined.txt'
    if os.path.exists(DATASET):
        G = prepare_graph(DATASET, seed=42)
        # Prueba rápida: candidatos = 30 nodos aleatorios
        np.random.seed(42)
        candidatos = list(np.random.choice(list(G.nodes()), 30, replace=False))
    else:
        print("Dataset no encontrado. Usando grafo dummy.")
        G = generate_dummy_graph(n=100, seed=42)
        candidatos = list(G.nodes())[:20]

    semillas, evals = phase2_sis_dynamic(
        G, candidatos,
        total_budget=50,
        time_limit=3,    # corto para prueba
        beta=0.02,
        delta=0.10,
        mc_simulations=5,
        verbose=True
    )

    print(f"\nInfluencia sostenida (T=20, MC=10):")
    influencia = compute_sustained_influence(
        G, semillas, T=20, beta=0.02, delta=0.10, mc_simulations=10
    )
    print(f"  IS(S) = {influencia:.2f} nodos activos promedio")
    print(f"\n[Verificación C1] Semillas únicas: "
          f"{len(semillas) == len(set(semillas))}")
