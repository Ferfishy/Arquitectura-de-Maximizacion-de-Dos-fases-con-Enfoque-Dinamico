import os
import random
import networkx as nx
import pandas as pd
from tqdm import tqdm

from graph_utils import prepare_graph
from phase1_celf import ic_model_spread

def extract_random_connected_subgraphs(G, num_subgraphs=5, size_min=50, size_max=100, seed=42):
    random.seed(seed)
    G_und = G.to_undirected()
    nodes = list(G.nodes())
    subgraphs = []
    
    for _ in range(num_subgraphs):
        n = random.randint(size_min, size_max)
        if n >= G.number_of_nodes():
            subgraphs.append(G.copy())
            continue
            
        root = random.choice(nodes)
        bfs_nodes = []
        visited = {root}
        queue = [root]
        
        while queue and len(bfs_nodes) < n:
            u = queue.pop(0)
            bfs_nodes.append(u)
            
            neighbors = list(G_und.neighbors(u))
            random.shuffle(neighbors)
            
            for v in neighbors:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
                    
        subgraph = G.subgraph(bfs_nodes).copy()
        subgraphs.append(subgraph)
        
    return subgraphs

def main():
    print("Iniciando generación del dataset multí-grafo...")
    
    # LISTA DE MULTIPLES DATASETS PARA GENERALIZAR LA GNN
    DATASETS = [
        # Datasets Ligeros a procesar:
        'datasets/musae_PTBR_edges.csv',
        'datasets/facebook_combined.txt',
        'datasets/Wiki-Vote.txt',
        'datasets/deezer_europe_edges.csv',
        'datasets/Email-Enron.txt',
        
        # Pesos Pesados
        'datasets/soc-Epinions1.txt',
        'datasets/Slashdot0811.txt'
    ]
    
    datos = []
    
    for dataset_file in DATASETS:
        base_name = os.path.splitext(os.path.basename(dataset_file))[0]
        output_file = f'dataset_{base_name}.csv'
        
        if os.path.exists(output_file):
            print(f"Ya existe {output_file}. Saltando generación...")
            continue
            
        if not os.path.exists(dataset_file):
            print(f"Error: No se encontró el dataset '{dataset_file}'. Saltando...")
            continue
            
        print(f"\n--- Procesando {dataset_file} ---")
        G = prepare_graph(dataset_file, seed=42)
        
        print("Calculando PageRank en el grafo completo...")
        pagerank_scores = nx.pagerank(G)
        
        print("Extrayendo subgrafos aleatorios...")
        subgrafos = extract_random_connected_subgraphs(G, num_subgraphs=5, size_min=50, size_max=100, seed=42)
        print(f"Se extrajeron {len(subgrafos)} subgrafos.")
        
        # Para grafos masivos (>50k nodos), reducimos las simulaciones MC para acelerar
        # la generación de datos sin perder demasiada representatividad.
        MC_SIMS = 20 if G.number_of_nodes() < 50000 else 5
        
        for idx, sg in enumerate(subgrafos):
            print(f"  Subgrafo {idx+1}/{len(subgrafos)} ({sg.number_of_nodes()} nodos)")
            
            for u in tqdm(sg.nodes(), desc=f"Nodos SG {idx+1}"):
                deg = G.degree(u)
                cost = G.nodes[u]['cost']
                pr = pagerank_scores[u]
                
                # Target: spread real simulado sobre el grafo completo (muy costoso)
                spread = ic_model_spread(G, [u], mc_simulations=MC_SIMS)
                
                datos.append({
                    'dataset_name': os.path.basename(dataset_file),
                    'node_id': u,
                    'subgraph_id': idx,
                    'degree': deg,
                    'cost': cost,
                    'pagerank': pr,
                    'target_spread': spread
                })
                
        # Guardar resultados de este dataset individualmente para no perder progreso
        df = pd.DataFrame(datos)
        base_name = os.path.splitext(os.path.basename(dataset_file))[0]
        output_file = f'dataset_{base_name}.csv'
        df.to_csv(output_file, index=False)
        print(f"\nDataset de {base_name} guardado en {output_file}")
        
        # Vaciar la lista para el siguiente grafo
        datos = []

if __name__ == '__main__':
    main()
