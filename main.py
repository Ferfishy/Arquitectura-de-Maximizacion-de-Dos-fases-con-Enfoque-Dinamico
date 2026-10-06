"""
main.py
=======
Punto de entrada del sistema de Maximización de Influencia en Dos Fases.

Ejecuta el pipeline completo sobre el dataset de Facebook SNAP:
  1. Carga y preprocesamiento del grafo.
  2. Fase 1: filtro voraz estático (IC + fast_mode).
  3. Fase 2: selección dinámica adaptativa (SIS con β y δ explícitos).
  4. Reporte de influencia sostenida del conjunto final.

Parámetros (alineados con la formalización del seminario):
  B=50, D=8, β=0.02, δ=0.10, T=20, seed=42.
"""

import time
import numpy as np

from graph_utils import prepare_graph, generate_dummy_graph
# from phase1_celf import phase1_celf
from fase1_gnn import phase1_gnn
from phase2_sis import phase2_sis_dynamic, compute_sustained_influence


# ─────────────────────────────────────────────────────────────────────────────
# Parámetros del experimento principal
# ─────────────────────────────────────────────────────────────────────────────

DATASET       = 'datasets/facebook_combined.txt'

# Reproducibilidad
SEED          = 42

# Fase 1
K_PRIME       = 20       # |V'|: candidatos seleccionados por Fase 1
MC_FASE1      = 50       # MC por nodo en el barrido rápido

# Fase 2
PRESUPUESTO   = 50       # B: presupuesto total disponible
DEADLINE      = 8        # D: número máximo de rondas de selección
BETA          = 0.02     # β: probabilidad de infección SIS (S→I)
DELTA         = 0.10     # δ: probabilidad de recuperación SIS (I→S)
MC_FASE2      = 10       # MC por evaluación Δ(u,t)

# Medición final
T_HORIZONTE   = 20       # horizonte T para influencia sostenida
MC_INFLUENCIA = 30       # MC para compute_sustained_influence


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import os

    np.random.seed(SEED)

    print("=" * 60)
    print("  MAXIMIZACIÓN DE INFLUENCIA EN DOS FASES")
    print("  Proyecto de Título")
    print(f"  Parámetros: B={PRESUPUESTO}, D={DEADLINE}, "
          f"β={BETA}, δ={DELTA}, T={T_HORIZONTE}, seed={SEED}")
    print("=" * 60)

    # ── Carga del grafo ───────────────────────────────────────────────────────
    if os.path.exists(DATASET):
        G = prepare_graph(DATASET, escala_costos=12.0, seed=SEED)
    else:
        print(f"\n[ADVERTENCIA] '{DATASET}' no encontrado.")
        print("  → Usando grafo Barabási-Albert de prueba (300 nodos).")
        G = generate_dummy_graph(n=300, m=4, seed=SEED)

    print(f"\n  Grafo: {G.number_of_nodes():,} nodos, "
          f"{G.number_of_edges():,} aristas")

    # ── Fase 1: Filtrado Rápido con GNN ───────────────────────────────────────
    t0 = time.perf_counter()

    candidatos = phase1_gnn(
        G,
        k_prime=K_PRIME,
        model_path='modelo_gnn.pth'
    )

    t1 = time.perf_counter()
    tiempo_fase1 = t1 - t0
    print(f"\n  [Fase 1] {len(candidatos)} candidatos extraídos "
          f"en {tiempo_fase1:.1f}s")

    # ── Fase 2: Selección Dinámica ─────────────────────────────────────────────
    t2 = time.perf_counter()

    semillas_finales, sis_evals = phase2_sis_dynamic(
        G,
        candidates=candidatos,
        total_budget=PRESUPUESTO,
        time_limit=DEADLINE,
        beta=BETA,
        delta=DELTA,
        mc_simulations=MC_FASE2,
        verbose=True
    )

    t3 = time.perf_counter()
    tiempo_fase2 = t3 - t2

    # ── Influencia Sostenida ───────────────────────────────────────────────────
    print(f"\n  Calculando influencia sostenida IS(S) "
          f"(T={T_HORIZONTE}, MC={MC_INFLUENCIA})...")
    influencia = compute_sustained_influence(
        G,
        seeds=semillas_finales,
        T=T_HORIZONTE,
        beta=BETA,
        delta=DELTA,
        mc_simulations=MC_INFLUENCIA
    )

    # ── Reporte Final ──────────────────────────────────────────────────────────
    costo_total = sum(G.nodes[u]['cost'] for u in semillas_finales)

    print("\n" + "=" * 60)
    print("  RESULTADOS FINALES")
    print("=" * 60)
    print(f"  Semillas seleccionadas ({len(semillas_finales)}): "
          f"{semillas_finales}")
    print(f"  Costo total invertido : {costo_total:.1f} / {PRESUPUESTO}")
    print(f"  Influencia sostenida  : IS(S) = {influencia:.2f} nodos activos")
    print(f"  Evaluaciones SIS      : {sis_evals}")
    print(f"  Tiempo Fase 1         : {tiempo_fase1:.1f}s")
    print(f"  Tiempo Fase 2         : {tiempo_fase2:.1f}s")
    print(f"  Tiempo TOTAL          : {tiempo_fase1 + tiempo_fase2:.1f}s")

    # Verificar invariante C1 (sin duplicados)
    assert len(semillas_finales) == len(set(semillas_finales)), \
        "ERROR: las semillas contienen duplicados (violación C1)."
    print(f"\n  [✓] Invariante C1: semillas únicas ({len(semillas_finales)} distintas)")
    print("=" * 60)


if __name__ == '__main__':
    main()
