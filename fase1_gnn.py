import os
import torch
import networkx as nx
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from graph_utils import prepare_graph

# ─────────────────────────────────────────────────────────────────────────────
# Arquitectura del Modelo GNN (misma estructura que el entrenamiento)
# ─────────────────────────────────────────────────────────────────────────────
class GNNRegresor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GNNRegresor, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.out = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Dropout no es estrictamente necesario en inferencia (ya que usamos eval()),
        # pero mantenemos la arquitectura idéntica
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.out(x)
        return x.squeeze()

# ─────────────────────────────────────────────────────────────────────────────
# Reemplazo de la Fase 1
# ─────────────────────────────────────────────────────────────────────────────
def phase1_gnn(G, k_prime: int = 25, model_path: str = 'modelo_gnn.pth') -> list:
    """
    Fase 1: Filtrado Rápido Basado en Inteligencia Artificial (GNN).
    
    Sustituye al CELF iterativo/voraz. Realiza una pasada Forward a través
    de la red neuronal para estimar el spread (Ganancia Marginal) de cada
    nodo de forma casi instantánea, seleccionando el top k'.
    
    Retorna: Lista de k' nodos candidatos.
    """
    print(f"\n{'='*55}")
    print(f"FASE 1: Filtrado Rápido con GNN (Inferencia)")
    print(f"  Nodos en G: {G.number_of_nodes():,}  |  k' objetivo: {k_prime}")
    print(f"{'='*55}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Preparación y extracción de características
    nodos_lista = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodos_lista)}
    num_nodes = len(nodos_lista)
    
    print("  [GNN] Calculando centralidad PageRank...")
    pagerank_scores = nx.pagerank(G)
    
    print("  [GNN] Construyendo matriz de características y normalizando...")
    x = torch.zeros((num_nodes, 3), dtype=torch.float)
    for n in nodos_lista:
        idx = node_to_idx[n]
        x[idx, 0] = G.degree(n)
        x[idx, 1] = G.nodes[n]['cost']
        x[idx, 2] = pagerank_scores[n]
        
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0)
    x = (x - x_mean) / (x_std + 1e-8)
    
    # Estructura de aristas
    edges = list(G.edges())
    edge_index = torch.tensor([[node_to_idx[u], node_to_idx[v]] for u, v in edges], dtype=torch.long).t().contiguous()
    
    data = Data(x=x, edge_index=edge_index).to(device)
    
    # 2. Carga del modelo
    model = GNNRegresor(in_channels=3, hidden_channels=32).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  [GNN] Pesos exitosamente cargados desde '{model_path}'")
    else:
        print(f"  [GNN] ADVERTENCIA: No se encontró '{model_path}'.")
        
    # 3. Modo de evaluación e Inferencia
    model.eval()
    print("  [GNN] Realizando inferencia sobre TODOS los nodos...")
    with torch.no_grad():
        preds = model(data.x, data.edge_index)
        
    preds_np = preds.cpu().numpy()
    
    # Asociar cada nodo original con su spread predicho
    node_preds = [(nodos_lista[i], preds_np[i]) for i in range(num_nodes)]
    
    # 4. Ordenar y seleccionar los mejores
    node_preds.sort(key=lambda item: item[1], reverse=True)
    top_k_preds = node_preds[:k_prime]
    candidatos = [n for n, p in top_k_preds]
    
    # 5. Impresión de resultados
    print(f"\n[Fase 1 GNN] Inferencia completada. Top {k_prime} seleccionados:")
    for i, (n, p) in enumerate(top_k_preds[:15]):  # Muestra los primeros 15 en consola para no saturar
        print(f"  {i+1:2d}. Nodo ID: {n:<5d} | Predicción de Spread: {p:.4f}")
    if k_prime > 15:
        print(f"  ... (y {k_prime - 15} nodos más)")
        
    return candidatos

# ─────────────────────────────────────────────────────────────────────────────
# Ejecución Directa (Prueba)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    DATASET_FILE = 'datasets/facebook_combined.txt'
    if os.path.exists(DATASET_FILE):
        G = prepare_graph(DATASET_FILE, seed=42)
        # Probamos la extracción
        candidatos = phase1_gnn(G, k_prime=25, model_path='modelo_gnn.pth')
        print(f"\nArray final de candidatos que se enviará a Fase 2: {candidatos}")
    else:
        print(f"Error: dataset '{DATASET_FILE}' no encontrado.")
