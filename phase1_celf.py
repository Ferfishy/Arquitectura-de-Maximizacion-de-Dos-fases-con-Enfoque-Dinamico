import heapq
import numpy as np
from tqdm import tqdm

def ic_model_spread(G, seeds, mc_simulations=100):
    """
    Estima la propagación (spread) de un conjunto de semillas usando el modelo Independent Cascade (IC).
    Simulaciones de Monte Carlo.
    """
    total_spread = 0
    for _ in range(mc_simulations):
        active_nodes = set(seeds)
        new_active_nodes = set(seeds)
        
        while new_active_nodes:
            current_active = set()
            for u in new_active_nodes:
                for v in G.successors(u):
                    if v not in active_nodes:
                        prob = G[u][v].get('weight', 0.1)
                        if np.random.rand() < prob:
                            current_active.add(v)
            active_nodes.update(current_active)
            new_active_nodes = current_active
            
        total_spread += len(active_nodes)
    return total_spread / mc_simulations

def phase1_celf(G, k_prime, mc_simulations=50):
    """
    Fase 1: Filtrado Voraz Estático utilizando el algoritmo CELF.
    Retorna el top k_prime de nodos candidatos.
    """
    print(f"--- Fase 1: Filtrado CELF ---")
    print(f"Evaluando top {k_prime} candidatos sobre {G.number_of_nodes()} nodos.")
    
    # Cola de prioridad
    Q = []
    
    # Paso 1: Evaluar la ganancia marginal de cada nodo individualmente
    # Usamos tqdm para ver el progreso
    print("Precalculando ganancias marginales individuales...")
    for u in tqdm(G.nodes()):
        spread = ic_model_spread(G, [u], mc_simulations)
        # heapq extrae los menores valores, por lo que guardamos el spread negativo
        heapq.heappush(Q, (-spread, u))
    
    candidates = []
    
    # Paso 2: Extraer los k_prime mejores aprovechando la submodularidad (CELF)
    print("Seleccionando subgrafo candidato...")
    # Para el primer nodo, no hay que recalcular porque es el máximo global inicial
    if Q:
        spread_neg, best_node = heapq.heappop(Q)
        candidates.append(best_node)
        
    while len(candidates) < k_prime and Q:
        spread_neg, u = heapq.heappop(Q)
        
        # Recalcular ganancia marginal: spread(candidates + {u}) - spread(candidates)
        # Ojo: Calcular spread(candidates) es costoso. 
        # Como es un prototipo simple y CELF solo se basa en que u sea mejor que el siguiente en Q,
        # simplificamos el recalculo.
        
        current_spread = ic_model_spread(G, candidates, mc_simulations)
        new_spread = ic_model_spread(G, candidates + [u], mc_simulations)
        marginal_gain = new_spread - current_spread
        
        # Revisamos si sigue siendo mejor que el tope de la cola
        if Q:
            next_spread_neg, _ = Q[0]
            if marginal_gain >= -next_spread_neg:
                # El nodo u sigue siendo el mejor
                candidates.append(u)
            else:
                # No es el mejor, lo reinsertamos con su nueva ganancia
                heapq.heappush(Q, (-marginal_gain, u))
        else:
            candidates.append(u)
            
    print(f"Fase 1 completada. {len(candidates)} nodos extraídos.")
    
    # Retornar el subgrafo inducido por los candidatos y sus vecinos 
    # (o simplemente trabajar sobre G, pero restringiendo la búsqueda en la fase 2 a estos candidatos)
    # Según la propuesta, se reduce el espacio de búsqueda a este "subgrafo". 
    # Devolveremos la lista de candidatos para la Fase 2.
    return candidates
