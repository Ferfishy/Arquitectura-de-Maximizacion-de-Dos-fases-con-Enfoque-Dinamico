import os
import torch
import pandas as pd
import networkx as nx
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

from graph_utils import prepare_graph

def create_data_object(dataset_file, df_subset):
    """
    Construye un objeto Data de PyTorch Geometric para un grafo específico,
    incluyendo sus máscaras de entrenamiento y validación basadas en los
    nodos que fueron etiquetados en el CSV.
    """
    G = prepare_graph(dataset_file, seed=42)
    nodos_lista = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodos_lista)}
    num_nodes = len(nodos_lista)
    
    pagerank_scores = nx.pagerank(G)
    
    x = torch.zeros((num_nodes, 3), dtype=torch.float)
    for n in nodos_lista:
        idx = node_to_idx[n]
        x[idx, 0] = G.degree(n)
        x[idx, 1] = G.nodes[n]['cost']
        x[idx, 2] = pagerank_scores[n]
        
    x_mean = x.mean(dim=0)
    x_std = x.std(dim=0)
    x = (x - x_mean) / (x_std + 1e-8)
        
    edges = list(G.edges())
    edge_index = torch.tensor([[node_to_idx[u], node_to_idx[v]] for u, v in edges], dtype=torch.long).t().contiguous()
    
    y = torch.zeros(num_nodes, dtype=torch.float)
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    for _, row in df_subset.iterrows():
        n = int(row['node_id'])
        if n in node_to_idx:
            idx = node_to_idx[n]
            y[idx] = float(row['target_spread'])
            mask[idx] = True
            
    labeled_indices = torch.nonzero(mask, as_tuple=True)[0].numpy()
    if len(labeled_indices) == 0:
        return None
        
    train_idx, test_idx = train_test_split(labeled_indices, test_size=0.2, random_state=42)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.test_mask = test_mask
    return data

# ─────────────────────────────────────────────────────────────────────────────
# Arquitectura del Modelo
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
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.out(x)
        return x.squeeze()

import glob

def main():
    print("1. Cargando datos tabulares multí-grafo...")
    csv_files = glob.glob('dataset_*.csv')
    if not csv_files:
        print("Error: No se encontraron archivos 'dataset_*.csv'. Corre el generador primero.")
        return
        
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)
    
    datasets_names = df['dataset_name'].unique()
    data_list = []
    
    print("2. Preparando objetos Data por cada red social...")
    for d_name in datasets_names:
        filepath = os.path.join('datasets', d_name)
        if os.path.exists(filepath):
            print(f" -> Construyendo Data para {d_name}...")
            df_sub = df[df['dataset_name'] == d_name]
            data_obj = create_data_object(filepath, df_sub)
            if data_obj is not None:
                data_list.append(data_obj)
        else:
            print(f" -> Advertencia: Archivo {filepath} no existe.")
            
    if not data_list:
        print("No se pudieron cargar datos.")
        return
        
    # DataLoader para empaquetar los grafos en batches
    loader = DataLoader(data_list, batch_size=1, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNRegresor(in_channels=3, hidden_channels=32).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    
    epochs = 200
    print(f"\n3 & 4. Iniciando entrenamiento de GNN sobre {len(data_list)} grafos...")
    print(f"Usando dispositivo: {device}")
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        total_test_loss = 0
        
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            
            # Loss solo en nodos de train de este grafo en particular
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            
            with torch.no_grad():
                model.eval()
                pred = model(data.x, data.edge_index)
                if data.test_mask.sum() > 0:
                    t_loss = criterion(pred[data.test_mask], data.y[data.test_mask]).item()
                    total_test_loss += t_loss
                model.train()
                
        avg_train_loss = total_train_loss / len(loader)
        avg_test_loss = total_test_loss / len(loader)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Época {epoch:03d}/{epochs} | Train MSE (Promedio): {avg_train_loss:.4f} | Test MSE: {avg_test_loss:.4f}")
            
    output_model_path = 'modelo_gnn.pth'
    torch.save(model.state_dict(), output_model_path)
    print(f"\n5. Entrenamiento finalizado. Modelo hiper-generalizado guardado en '{output_model_path}'.")

if __name__ == '__main__':
    main()
