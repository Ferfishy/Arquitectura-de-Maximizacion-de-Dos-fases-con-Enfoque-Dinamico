"""
experiments.py
==============
Script de experimentos comparativos para la Arquitectura de Maximización
de Influencia en Dos Fases (seminario de título).

Experimentos implementados:
  1. Dos-fases vs. Baseline: compara influencia sostenida, evaluaciones SIS
     y tiempo de ejecución sobre el grafo completo.
  2. Escalabilidad: repite la comparativa sobre submuestras crecientes de la
     red de Facebook (200, 500, 1000, 2000, 4039 nodos).

Métricas reportadas por experimento:
  - Influencia sostenida IS(S): promedio de activos SIS en ventana T.
  - Evaluaciones SIS: número total de llamadas a sis_model_spread_step
    (mide carga computacional de la selección).
  - Tiempo de ejecución (segundos).

INVARIANTE preservada en todos los experimentos: la difusión SIS y el cálculo
de influencia sostenida siempre se realizan sobre el grafo COMPLETO G;
la Fase 1 sólo reduce el conjunto de candidatos seleccionables.
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')   # backend sin pantalla (compatible con todos los entornos)
import matplotlib.pyplot as plt

from graph_utils import prepare_graph, sample_connected_subgraph
from phase1_celf import phase1_celf
from phase2_sis import phase2_sis_dynamic, compute_sustained_influence


# ─────────────────────────────────────────────────────────────────────────────
# Parámetros globales del experimento
# ─────────────────────────────────────────────────────────────────────────────

SEED        = 42       # reproducibilidad global
DATASET     = 'facebook_combined.txt'

# Fase 1
K_PRIME     = 150      # candidatos seleccionados por la Fase 1 (≈3.7% de 4039)
MC_FASE1    = 50       # MC por nodo en fast_mode

# Fase 2
PRESUPUESTO = 50       # B
DEADLINE    = 8        # D
BETA        = 0.02     # probabilidad de infección SIS (S→I)
DELTA       = 0.10     # probabilidad de recuperación SIS (I→S)
MC_FASE2    = 10       # MC por evaluación Δ(u,t)

# Medición de influencia sostenida
T_HORIZONTE = 20       # ventana temporal
MC_INFLUENCIA = 30     # MC para compute_sustained_influence

# Experimento de escalabilidad
TAMANIOS_ESCALA = [200, 500, 1000, 2000, 4039]
K_PRIME_ESCALA  = 50   # candidatos para submuestras pequeñas


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ejecutar sistema dos-fases
# ─────────────────────────────────────────────────────────────────────────────

def run_two_phase(G,
                  k_prime: int = K_PRIME,
                  B: float = PRESUPUESTO,
                  D: int = DEADLINE,
                  beta: float = BETA,
                  delta: float = DELTA,
                  T: int = T_HORIZONTE,
                  mc_p1: int = MC_FASE1,
                  mc_p2: int = MC_FASE2,
                  mc_inf: int = MC_INFLUENCIA,
                  verbose: bool = True) -> dict:
    """
    Ejecuta el pipeline completo Dos-Fases sobre el grafo G.

    Returns
    -------
    dict con:
      'semillas'       : lista de semillas seleccionadas.
      'influencia'     : IS(S) — influencia sostenida sobre G completo.
      'sis_evals'      : evaluaciones SIS en Fase 2.
      'tiempo_fase1'   : segundos de Fase 1.
      'tiempo_fase2'   : segundos de Fase 2.
      'tiempo_total'   : segundos totales.
      'costo_total'    : suma de c(u) para u ∈ S.
    """
    resultado = {}

    # ── Fase 1 ──────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    candidatos = phase1_celf(G, k_prime=k_prime, mc_simulations=mc_p1,
                              fast_mode=True)
    t1 = time.perf_counter()
    resultado['tiempo_fase1'] = t1 - t0

    if verbose:
        print(f"\n[Dos-fases] Fase 1 completada en {resultado['tiempo_fase1']:.1f}s "
              f"→ {len(candidatos)} candidatos.")

    # ── Fase 2 ──────────────────────────────────────────────────────────────
    t2 = time.perf_counter()
    semillas, sis_evals = phase2_sis_dynamic(
        G, candidatos,
        total_budget=B, time_limit=D,
        beta=beta, delta=delta,
        mc_simulations=mc_p2,
        verbose=verbose
    )
    t3 = time.perf_counter()
    resultado['tiempo_fase2'] = t3 - t2
    resultado['tiempo_total'] = t3 - t0

    # ── Influencia sostenida (sobre G completo) ──────────────────────────────
    if verbose:
        print(f"\n[Dos-fases] Calculando influencia sostenida "
              f"(T={T}, MC={mc_inf})...")
    influencia = compute_sustained_influence(
        G, semillas, T=T, beta=beta, delta=delta, mc_simulations=mc_inf
    )

    resultado['semillas']    = semillas
    resultado['influencia']  = influencia
    resultado['sis_evals']   = sis_evals
    resultado['costo_total'] = sum(G.nodes[u]['cost'] for u in semillas)

    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ejecutar baseline (sin filtro de Fase 1)
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(G,
                 B: float = PRESUPUESTO,
                 D: int = DEADLINE,
                 beta: float = BETA,
                 delta: float = DELTA,
                 T: int = T_HORIZONTE,
                 mc_p2: int = MC_FASE2,
                 mc_inf: int = MC_INFLUENCIA,
                 verbose: bool = True) -> dict:
    """
    Baseline: ejecuta la Fase 2 directamente sobre TODOS los nodos del grafo
    (sin filtro de Fase 1). Mide la misma influencia sostenida y evaluaciones.

    Esta comparación permite cuantificar:
      - La pérdida de calidad (si alguna) del filtro de Fase 1.
      - El ahorro computacional de la Fase 1 en términos de evaluaciones SIS.

    Returns
    -------
    dict con las mismas claves que run_two_phase.
    """
    resultado = {}

    if verbose:
        print(f"\n{'='*55}")
        print(f"BASELINE: Selección dinámica sobre TODOS los nodos")
        print(f"  |V| = {G.number_of_nodes():,}  |  "
              f"B={B}  |  D={D}  |  β={beta}  |  δ={delta}")
        print(f"{'='*55}")

    todos_los_nodos = list(G.nodes())

    t0 = time.perf_counter()
    semillas, sis_evals = phase2_sis_dynamic(
        G, todos_los_nodos,
        total_budget=B, time_limit=D,
        beta=beta, delta=delta,
        mc_simulations=mc_p2,
        verbose=verbose
    )
    t1 = time.perf_counter()

    resultado['tiempo_fase1'] = 0.0   # sin Fase 1
    resultado['tiempo_fase2'] = t1 - t0
    resultado['tiempo_total'] = t1 - t0

    if verbose:
        print(f"\n[Baseline] Calculando influencia sostenida "
              f"(T={T}, MC={mc_inf})...")
    influencia = compute_sustained_influence(
        G, semillas, T=T, beta=beta, delta=delta, mc_simulations=mc_inf
    )

    resultado['semillas']    = semillas
    resultado['influencia']  = influencia
    resultado['sis_evals']   = sis_evals
    resultado['costo_total'] = sum(G.nodes[u]['cost'] for u in semillas)

    return resultado


# ─────────────────────────────────────────────────────────────────────────────
# 3. Tabla comparativa
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(res_dos_fases: dict,
                           res_baseline: dict,
                           n_nodos: int = None) -> None:
    """
    Imprime una tabla comparativa de calidad y eficiencia.

    Métricas mostradas:
      - Semillas seleccionadas (|S|)
      - Costo total invertido
      - Influencia sostenida IS(S)
      - Evaluaciones SIS (carga computacional)
      - Tiempo de ejecución total
    """
    cabecera = f"{'Métrica':<30} {'Dos-Fases':>12} {'Baseline':>12} {'Δ%':>10}"
    separador = "─" * len(cabecera)

    print(f"\n{'='*60}")
    if n_nodos:
        print(f"  TABLA COMPARATIVA  |  Red: {n_nodos:,} nodos")
    else:
        print(f"  TABLA COMPARATIVA")
    print(f"{'='*60}")
    print(cabecera)
    print(separador)

    def fila(metrica, val_df, val_bl, fmt=".2f"):
        try:
            delta_pct = ((val_df - val_bl) / abs(val_bl)) * 100 if val_bl != 0 else 0
            delta_str = f"{delta_pct:+.1f}%"
        except Exception:
            delta_str = "  —"
        print(f"  {metrica:<28} {val_df:>12{fmt}} {val_bl:>12{fmt}} {delta_str:>10}")

    fila("Semillas |S|",
         len(res_dos_fases['semillas']),
         len(res_baseline['semillas']), fmt=".0f")

    fila("Costo total invertido",
         res_dos_fases['costo_total'],
         res_baseline['costo_total'], fmt=".1f")

    fila("Influencia sostenida IS(S)",
         res_dos_fases['influencia'],
         res_baseline['influencia'], fmt=".2f")

    fila("Evaluaciones SIS",
         res_dos_fases['sis_evals'],
         res_baseline['sis_evals'], fmt=".0f")

    fila("Tiempo total (s)",
         res_dos_fases['tiempo_total'],
         res_baseline['tiempo_total'], fmt=".1f")

    print(separador)
    print(f"\n  Tiempo Fase 1 (dos-fases): {res_dos_fases['tiempo_fase1']:.1f}s")
    print(f"  Semillas dos-fases : {res_dos_fases['semillas']}")
    print(f"  Semillas baseline  : {res_baseline['semillas']}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Experimento de escalabilidad
# ─────────────────────────────────────────────────────────────────────────────

def scalability_experiment(G_completo,
                            tamanios: list = None,
                            k_prime: int = K_PRIME_ESCALA,
                            B: float = PRESUPUESTO,
                            D: int = DEADLINE,
                            beta: float = BETA,
                            delta: float = DELTA,
                            T: int = T_HORIZONTE,
                            mc_p1: int = 20,
                            mc_p2: int = 5,
                            mc_inf: int = 15) -> dict:
    """
    Experimento de escalabilidad: corre dos-fases y baseline sobre subgrafos
    conexos de tamaños crecientes, registrando influencia y tiempo.

    Para tamaños pequeños se usa fast_mode=True con mc_p1 reducido para
    mantener el experimento ejecutable en tiempo razonable.

    Parameters
    ----------
    G_completo : grafo completo de Facebook preprocesado.
    tamanios   : lista de tamaños de subgrafos a evaluar.
    ...        : parámetros del modelo (ver defaults en la cabecera).

    Returns
    -------
    dict con listas paralelas por tamaño:
      'tamanios', 'inf_dos_fases', 'inf_baseline',
      'time_dos_fases', 'time_baseline',
      'evals_dos_fases', 'evals_baseline'
    """
    if tamanios is None:
        tamanios = TAMANIOS_ESCALA

    registros = {
        'tamanios'       : [],
        'inf_dos_fases'  : [],
        'inf_baseline'   : [],
        'time_dos_fases' : [],
        'time_baseline'  : [],
        'evals_dos_fases': [],
        'evals_baseline' : [],
    }

    print(f"\n{'#'*60}")
    print(f"  EXPERIMENTO DE ESCALABILIDAD")
    print(f"  Tamaños: {tamanios}")
    print(f"{'#'*60}\n")

    for tam in tamanios:
        print(f"\n{'─'*50}")
        print(f"  Submuestra: {tam} nodos")
        print(f"{'─'*50}")

        # Extraer subgrafo conexo
        G_sub = sample_connected_subgraph(G_completo, tam, seed=SEED)
        n_real = G_sub.number_of_nodes()
        k_sub = min(k_prime, max(10, n_real // 5))   # k' ≤ 20% del subgrafo

        # ── Dos-fases ───────────────────────────────────────────────────────
        print(f"  [Dos-fases] k'={k_sub}, MC_p1={mc_p1}, MC_p2={mc_p2}")
        res_df = run_two_phase(
            G_sub,
            k_prime=k_sub, B=B, D=D,
            beta=beta, delta=delta, T=T,
            mc_p1=mc_p1, mc_p2=mc_p2, mc_inf=mc_inf,
            verbose=False
        )

        # ── Baseline ────────────────────────────────────────────────────────
        print(f"  [Baseline] |V|={n_real}, MC_p2={mc_p2}")
        res_bl = run_baseline(
            G_sub,
            B=B, D=D,
            beta=beta, delta=delta, T=T,
            mc_p2=mc_p2, mc_inf=mc_inf,
            verbose=False
        )

        # Registrar
        registros['tamanios'].append(n_real)
        registros['inf_dos_fases'].append(res_df['influencia'])
        registros['inf_baseline'].append(res_bl['influencia'])
        registros['time_dos_fases'].append(res_df['tiempo_total'])
        registros['time_baseline'].append(res_bl['tiempo_total'])
        registros['evals_dos_fases'].append(res_df['sis_evals'])
        registros['evals_baseline'].append(res_bl['sis_evals'])

        print(f"  IS dos-fases={res_df['influencia']:.2f}  "
              f"IS baseline={res_bl['influencia']:.2f}  |  "
              f"t_df={res_df['tiempo_total']:.1f}s  "
              f"t_bl={res_bl['tiempo_total']:.1f}s")

    return registros


# ─────────────────────────────────────────────────────────────────────────────
# 5. Graficado de resultados de escalabilidad
# ─────────────────────────────────────────────────────────────────────────────

def plot_scalability(registros: dict,
                     output_path: str = 'escalabilidad.png') -> None:
    """
    Genera un gráfico de 3 paneles con los resultados de escalabilidad:
      - Panel superior   : Influencia sostenida IS(S) vs. tamaño de red.
      - Panel central    : Tiempo de ejecución vs. tamaño de red.
      - Panel inferior   : Evaluaciones SIS vs. tamaño de red.
    """
    tamanios = registros['tamanios']
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), dpi=120)
    fig.suptitle(
        'Escalabilidad: Dos-Fases vs. Baseline\n'
        'Maximización de Influencia con SIS (β=0.02, δ=0.10)',
        fontsize=13, fontweight='bold', y=0.98
    )

    colores = {'df': '#2563EB', 'bl': '#DC2626'}
    estilos  = {'df': 'o-',     'bl': 's--'}

    # ── Panel 1: Influencia sostenida ────────────────────────────────────────
    ax = axes[0]
    ax.plot(tamanios, registros['inf_dos_fases'],
            estilos['df'], color=colores['df'],
            label='Dos-Fases', linewidth=2, markersize=6)
    ax.plot(tamanios, registros['inf_baseline'],
            estilos['bl'], color=colores['bl'],
            label='Baseline', linewidth=2, markersize=6)
    ax.set_ylabel('IS(S) — nodos activos promedio', fontsize=10)
    ax.set_title('Influencia Sostenida', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # ── Panel 2: Tiempo de ejecución ─────────────────────────────────────────
    ax = axes[1]
    ax.plot(tamanios, registros['time_dos_fases'],
            estilos['df'], color=colores['df'],
            label='Dos-Fases', linewidth=2, markersize=6)
    ax.plot(tamanios, registros['time_baseline'],
            estilos['bl'], color=colores['bl'],
            label='Baseline', linewidth=2, markersize=6)
    ax.set_ylabel('Tiempo (segundos)', fontsize=10)
    ax.set_title('Tiempo de Ejecución Total', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    # ── Panel 3: Evaluaciones SIS ─────────────────────────────────────────────
    ax = axes[2]
    ax.plot(tamanios, registros['evals_dos_fases'],
            estilos['df'], color=colores['df'],
            label='Dos-Fases', linewidth=2, markersize=6)
    ax.plot(tamanios, registros['evals_baseline'],
            estilos['bl'], color=colores['bl'],
            label='Baseline', linewidth=2, markersize=6)
    ax.set_xlabel('Tamaño de la red (nodos, escala log)', fontsize=10)
    ax.set_ylabel('Evaluaciones SIS', fontsize=10)
    ax.set_title('Carga Computacional (evaluaciones SIS)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, bbox_inches='tight')
    print(f"\n[Gráfico] Guardado en '{output_path}'")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Tabla de escalabilidad
# ─────────────────────────────────────────────────────────────────────────────

def print_scalability_table(registros: dict) -> None:
    """Imprime tabla de escalabilidad en formato texto."""
    col = 10
    print(f"\n{'='*75}")
    print(f"  TABLA DE ESCALABILIDAD")
    print(f"{'='*75}")
    cab = (f"{'|V|':>{col}} {'IS_df':>{col}} {'IS_bl':>{col}} "
           f"{'t_df(s)':>{col}} {'t_bl(s)':>{col}} "
           f"{'evals_df':>{col}} {'evals_bl':>{col}}")
    print(cab)
    print("─" * 75)

    for i, tam in enumerate(registros['tamanios']):
        print(f"  {tam:>{col-2},} "
              f"{registros['inf_dos_fases'][i]:>{col}.2f} "
              f"{registros['inf_baseline'][i]:>{col}.2f} "
              f"{registros['time_dos_fases'][i]:>{col}.1f} "
              f"{registros['time_baseline'][i]:>{col}.1f} "
              f"{registros['evals_dos_fases'][i]:>{col}} "
              f"{registros['evals_baseline'][i]:>{col}}")

    print(f"{'='*75}")
    print(f"  IS = Influencia Sostenida (nodos activos promedio)")
    print(f"  df = Dos-Fases  |  bl = Baseline")
    print(f"{'='*75}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Bloque principal
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os

    np.random.seed(SEED)

    print(f"\n{'#'*60}")
    print(f"  EXPERIMENTOS: Maximización de Influencia en Dos Fases")
    print(f"  Dataset: {DATASET}")
    print(f"  β={BETA}  δ={DELTA}  B={PRESUPUESTO}  D={DEADLINE}  T={T_HORIZONTE}")
    print(f"  k'={K_PRIME}  seed={SEED}")
    print(f"{'#'*60}\n")

    # ── Cargar grafo ─────────────────────────────────────────────────────────
    if not os.path.exists(DATASET):
        print(f"[ADVERTENCIA] '{DATASET}' no encontrado.")
        print("  → Usando grafo Barabási-Albert de prueba (500 nodos).")
        from graph_utils import generate_dummy_graph
        G = generate_dummy_graph(n=500, m=3, seed=SEED)
        # Ajustar parámetros para el grafo pequeño
        K_PRIME_EXPERIMENTO = 30
        TAMANIOS_EXPERIMENTO = [50, 100, 200, 300, 500]
    else:
        G = prepare_graph(DATASET, seed=SEED)
        K_PRIME_EXPERIMENTO = K_PRIME
        TAMANIOS_EXPERIMENTO = TAMANIOS_ESCALA

    # ── Experimento 1: Dos-Fases vs. Baseline sobre G completo ──────────────
    print(f"\n{'─'*60}")
    print(f"  EXPERIMENTO 1: Dos-Fases vs. Baseline (grafo completo)")
    print(f"{'─'*60}")

    print("\n  → Ejecutando sistema DOS-FASES...")
    res_df = run_two_phase(
        G,
        k_prime=K_PRIME_EXPERIMENTO,
        B=PRESUPUESTO, D=DEADLINE,
        beta=BETA, delta=DELTA,
        T=T_HORIZONTE,
        mc_p1=MC_FASE1, mc_p2=MC_FASE2,
        mc_inf=MC_INFLUENCIA,
        verbose=True
    )

    print("\n  → Ejecutando BASELINE (todos los nodos)...")
    res_bl = run_baseline(
        G,
        B=PRESUPUESTO, D=DEADLINE,
        beta=BETA, delta=DELTA,
        T=T_HORIZONTE,
        mc_p2=MC_FASE2,
        mc_inf=MC_INFLUENCIA,
        verbose=True
    )

    print_comparison_table(res_df, res_bl, n_nodos=G.number_of_nodes())

    # ── Experimento 2: Escalabilidad ─────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  EXPERIMENTO 2: Escalabilidad")
    print(f"{'─'*60}")

    registros = scalability_experiment(
        G,
        tamanios=TAMANIOS_EXPERIMENTO,
        k_prime=K_PRIME_ESCALA,
        B=PRESUPUESTO, D=DEADLINE,
        beta=BETA, delta=DELTA, T=T_HORIZONTE,
        mc_p1=20, mc_p2=5, mc_inf=15
    )

    print_scalability_table(registros)
    plot_scalability(registros, output_path='escalabilidad.png')

    print("\n¡Experimentos completados!")
    print("  Resultado guardado en: escalabilidad.png")
