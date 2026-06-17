import numpy as np
from tqdm import tqdm

def sis_model_spread_step(G, active_nodes, recovery_rate=0.3):
    """
    Ejecuta un paso de tiempo del modelo SIS.
    active_nodes: set de nodos infectados (I)
    Retorna el nuevo set de nodos activos.
    """
    new_active = set(active_nodes)
    
    # Propagación (S -> I)
    current_active = set(active_nodes)
    for u in current_active:
        for v in G.successors(u):
            if v not in current_active:
                prob = G[u][v].get('weight', 0.1)
                if np.random.rand() < prob:
                    new_active.add(v)
                    
    # Recuperación (I -> S)
    recovered = set()
    for u in current_active:
        if np.random.rand() < recovery_rate:
            recovered.add(u)
            
    # El nuevo estado es: los que estaban activos + los nuevos infectados - los recuperados
    new_active.difference_update(recovered)
    return new_active

def estimate_dynamic_marginal_gain(G, u, current_active_nodes, remaining_time, mc_simulations=10, recovery_rate=0.3):
    """
    Estima la ganancia marginal dinámica de activar el nodo u en el instante actual.
    Se proyecta la propagación en los instantes de tiempo restantes.
    """
    total_gain = 0
    for _ in range(mc_simulations):
        # Simulación SIN u
        active_without_u = set(current_active_nodes)
        spread_without_u = 0
        
        # Simulación CON u
        active_with_u = set(current_active_nodes)
        active_with_u.add(u)
        spread_with_u = 0
        
        for t in range(remaining_time):
            # Paso sin u
            active_without_u = sis_model_spread_step(G, active_without_u, recovery_rate)
            spread_without_u += len(active_without_u)
            
            # Paso con u
            active_with_u = sis_model_spread_step(G, active_with_u, recovery_rate)
            spread_with_u += len(active_with_u)
            
        total_gain += (spread_with_u - spread_without_u)
        
    return total_gain / mc_simulations

def phase2_sis_dynamic(G, candidates, total_budget, time_limit, recovery_rate=0.3):
    """
    Fase 2: Evaluación Dinámica Adaptativa bajo Restricciones
    Selecciona nodos considerando costo y presupuesto en un entorno dinámico (SIS).
    """
    print(f"--- Fase 2: Maximización Adaptativa bajo Restricciones ---")
    print(f"Presupuesto: {total_budget}, Tiempo Límite: {time_limit}")
    
    budget_actual = total_budget
    current_active_nodes = set()
    selected_seeds = []
    
    # El tiempo avanza en cada iteración de selección
    for t in range(time_limit):
        print(f"\n[Tiempo t={t}] Presupuesto restante: {budget_actual}")
        
        best_node = None
        best_ratio = -1
        
        # Filtrar candidatos que aún podemos pagar y no están activos
        affordable_candidates = [v for v in candidates if G.nodes[v]['cost'] <= budget_actual and v not in current_active_nodes]
        
        if not affordable_candidates:
            print("No hay candidatos asequibles o presupuesto agotado.")
            break
            
        print(f"Evaluando {len(affordable_candidates)} candidatos rentables...")
        for u in affordable_candidates:
            cost = G.nodes[u]['cost']
            
            # Ganancia esperada proyectada en el tiempo restante (D - t)
            marginal_gain = estimate_dynamic_marginal_gain(G, u, current_active_nodes, time_limit - t, mc_simulations=10, recovery_rate=recovery_rate)
            
            # Razón Beneficio-Costo
            ratio = marginal_gain / cost
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_node = u
                
        if best_node is not None and best_ratio > 0:
            print(f"-> Nodo seleccionado: {best_node} (Costo: {G.nodes[best_node]['cost']}, Ratio: {best_ratio:.2f})")
            selected_seeds.append(best_node)
            current_active_nodes.add(best_node)
            budget_actual -= G.nodes[best_node]['cost']
            
            # Opcional: Avanzar un instante el estado real de la red
            current_active_nodes = sis_model_spread_step(G, current_active_nodes, recovery_rate)
            print(f"Usuarios activos actualmente (aproximado): {len(current_active_nodes)}")
        else:
            print("Ningún nodo ofreció ganancia positiva. Deteniendo selección.")
            break
            
    print(f"\nFase 2 completada. Semillas finales seleccionadas: {selected_seeds}")
    return selected_seeds
