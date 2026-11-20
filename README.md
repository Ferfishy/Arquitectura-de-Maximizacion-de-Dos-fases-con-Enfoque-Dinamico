# Arquitectura de Maximización de Influencia en Dos Fases

Repositorio oficial del Proyecto de Título. Este software implementa una arquitectura híbrida (**Deep Learning sobre Grafos + Optimización Submodular Vectorizada**) para resolver el problema de **Maximización de Influencia Sostenida** bajo restricciones de presupuesto ($B$) y horizonte de decisión ($D$) en redes complejas de gran escala.

---

## 🧠 Descripción de la Arquitectura

El problema clásico de Maximización de Influencia es $\text{NP-Hard}$. Evaluar la ganancia marginal de todos los nodos de una red mediante simulaciones de Monte Carlo en un entorno dinámico resulta computacionalmente inviable a medida que el grafo crece.

Esta arquitectura desacopla el problema en **dos fases complementarias**:

1. **Fase 1 — Filtrado Rápido Topológico mediante GNN (`fase1_gnn.py`):**
   * Utiliza una red neuronal de grafos basada en convoluciones **GraphSAGE** (`SAGEConv` de *PyTorch Geometric*).
   * Extrae y normaliza localmente por grafo 3 características estructurales de cada nodo: **Grado**, **Costo de activación** y **Centralidad PageRank**.
   * Predice en una sola pasada de inferencia matricial ($< 1\text{ s}$ para redes de miles de nodos) el potencial de esparcimiento (*Spread*) de cada nodo y filtra el grafo completo $V$ a un subconjunto reducido de candidatos élite $V'$ de tamaño $k' \ll |V|$.
   * El modelo (`modelo_gnn.pth`) fue entrenado de forma inductiva y multí-grafo sobre **7 redes reales de topologías heterogéneas** utilizando *Ground Truth* generado con el modelo *Independent Cascade (IC)*.

2. **Fase 2 — Selección Dinámica Adaptativa con Modelo SIS Vectorizado (`phase2_sis.py`):**
   * Opera exclusivamente sobre los $k'$ candidatos seleccionados por la Fase 1, pero **simulando la propagación sobre el grafo completo $G$** (preservando la invariante topológica).
   * Modela la difusión como un proceso epidémico **Susceptible-Infectado-Susceptible (SIS)** con tasa de infección $\beta$ y tasa de recuperación $\delta$, implementado mediante álgebra lineal dispersa (`scipy.sparse.csr_matrix`).
   * En cada paso de decisión $t \in \{0, \dots, D-1\}$, selecciona el nodo factible $u \in V' \setminus S$ que maximiza el ratio de ganancia marginal dinámica sobre costo:
     $$\text{ratio}(u, t) = \frac{\Delta(u \mid X_t, D - t)}{c(u)}$$
     sujeto a la restricción presupuestaria $\sum_{s \in S} c(s) \le B$.

---

## 📂 Estructura del Proyecto

```text
├── datasets/                  # Redes sociales y grafos de entrada (.txt y .csv)
├── graph_utils.py             # Carga unificada de grafos (TXT/CSV), pesos WC y costos
├── generador_dataset.py       # Extracción de subgrafos BFS y generación de Ground Truth (IC)
├── entrenamiento_gnn.py       # Entrenamiento multí-grafo de la red GraphSAGE (PyTorch Geometric)
├── modelo_gnn.pth             # Pesos pre-entrenados de la GNN universal (7 datasets)
├── fase1_gnn.py               # Fase 1: Inferencia rápida con GNN y extracción de Top-k' candidatos
├── phase1_celf.py             # Fase 1 (Legado): Filtro voraz clásico IC / CELF
├── phase2_sis.py              # Fase 2: Selección dinámica SIS vectorizada y cálculo de IS(S)
├── main.py                    # Punto de entrada principal para ejecutar la arquitectura
└── experiments.py             # Banco de pruebas comparativo (Dos-Fases vs. Baseline y Escalabilidad)
```

---

## 📊 Datasets Soportados

El cargador unificado (`graph_utils.py`) soporta listas de aristas tanto en formato `.txt` (separadas por espacios/tabulaciones) como `.csv`. La GNN universal fue entrenada y validada con redes del repositorio [Stanford SNAP](https://snap.stanford.edu/data/):

| Dataset | Archivo | Categoría / Topología | Nodos ($|V|$) | Aristas Dirigidas ($|E|$) |
| :--- | :--- | :--- | :---: | :---: |
| **Twitch PTBR** | `musae_PTBR_edges.csv` | Red de Streaming | 1,912 | 62,598 |
| **Facebook** | `facebook_combined.txt` | Red Social (Círculos) | 4,039 | 176,468 |
| **Twitch FR** | `musae_FR_edges.csv` | Red de Streaming | 6,549 | 225,332 |
| **Wiki-Vote** | `Wiki-Vote.txt` | Red de Votación | 7,115 | 201,524 |
| **Deezer Europe** | `deezer_europe_edges.csv` | Red Musical de Amistad | 28,281 | 185,504 |
| **Email-Enron** | `Email-Enron.txt` | Red de Comunicaciones | 36,692 | 367,662 |
| **soc-Epinions1** | `soc-Epinions1.txt` | Red de Confianza Masiva | 75,879 | 811,480 |
| **Slashdot0811** | `Slashdot0811.txt` | Foro Tecnológico Masivo | 77,360 | 1,015,667 |

*(Para ejecutar un dataset, colócalo dentro de la carpeta `datasets/`).*

---

## ⚙️ Requisitos e Instalación

El proyecto requiere **Python 3.10+** y las siguientes librerías principales:

```bash
pip install numpy scipy pandas networkx matplotlib tqdm scikit-learn torch torch-geometric
```

---

## 🚀 Uso y Ejecución

### 1. Ejecutar el Pipeline Principal (`main.py`)
Ejecuta la Fase 1 (GNN) + Fase 2 (SIS dinámico) y reporta las semillas seleccionadas, el costo invertido, la **Influencia Sostenida $IS(S)$** y los tiempos de cómputo:

```bash
python3 main.py
```
> **Para cambiar de grafo:** Edita la variable `DATASET` en la línea 29 de `main.py` apuntando al archivo deseado dentro de `datasets/`.

### 2. Ejecutar Experimentos Comparativos y Escalabilidad (`experiments.py`)
Compara el rendimiento de la arquitectura **Dos-Fases** frente al **Baseline** (selección dinámica sobre todos los nodos sin filtro de Fase 1) y genera las curvas de escalabilidad en `escalabilidad.png`:

```bash
python3 experiments.py
```

### 3. Re-entrenar la GNN desde Cero (Opcional)
Si deseas agregar nuevos grafos al entrenamiento inductivo de la Fase 1:
1. Añade la ruta del grafo a la lista `DATASETS` en `generador_dataset.py` y ejecuta:
   ```bash
   python3 generador_dataset.py
   ```
2. Entrena el modelo unificado (que cargará todos los archivos `dataset_*.csv` generados y actualizará `modelo_gnn.pth`):
   ```bash
   python3 entrenamiento_gnn.py
   ```
