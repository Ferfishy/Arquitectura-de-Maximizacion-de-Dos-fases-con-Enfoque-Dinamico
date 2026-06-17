import time
from graph_utils import generate_dummy_graph, prepare_graph
from phase1_celf import phase1_celf
from phase2_sis import phase2_sis_dynamic

def main():
    print("=====================================================")
    print("PROTOTIPO DE MAXIMIZACIÓN DE INFLUENCIA EN DOS FASES")
    print("=====================================================")
    
    # Configuración de prueba
    # Cargar el dataset real
    G = prepare_graph('facebook_combined.txt')
    
    # Parámetros Fase 1
    K_PRIME = 30 # Reducir el espacio a los mejores nodos
    MC_SIMULATIONS_P1 = 10 # Reducido para acelerar la ejecución en un grafo grande
    
    # Parámetros Fase 2
    TOTAL_BUDGET = 50
    TIME_LIMIT = 5
    RECOVERY_RATE = 0.4
    
    # Ejecución Fase 1
    start_p1 = time.time()
    candidates = phase1_celf(G, k_prime=K_PRIME, mc_simulations=MC_SIMULATIONS_P1)
    end_p1 = time.time()
    
    print(f"\nCandidatos Fase 1: {candidates}")
    print(f"Tiempo Fase 1: {end_p1 - start_p1:.2f} segundos\n")
    
    # Ejecución Fase 2
    start_p2 = time.time()
    final_seeds = phase2_sis_dynamic(G, candidates, TOTAL_BUDGET, TIME_LIMIT, RECOVERY_RATE)
    end_p2 = time.time()
    
    print("=====================================================")
    print("RESULTADOS FINALES")
    print("=====================================================")
    print(f"Semillas Seleccionadas: {final_seeds}")
    costo_total = sum(G.nodes[u]['cost'] for u in final_seeds)
    print(f"Costo Total Invertido: {costo_total} / {TOTAL_BUDGET}")
    print(f"Tiempo de ejecución Fase 2: {end_p2 - start_p2:.2f} segundos")
    print("=====================================================")

if __name__ == "__main__":
    main()
