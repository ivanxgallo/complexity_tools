# point_process.py

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from tqdm.notebook import tqdm


class PointProcessNetwork:
    def __init__(self, dim=2, directed=False):
        self.dim = dim
        self.events = None
        self.grid = None
        self.graph = nx.DiGraph() if directed else nx.Graph()


    def load_events_from_df(self, df, time_col='t', space_cols=None):
        """
        Carga eventos desde un DataFrame.
        """
        if space_cols is None:
            space_cols = [col for col in df.columns if col != time_col]
        assert len(space_cols) == self.dim, "Dimensión del espacio no coincide"
        self.events = df[[time_col] + space_cols].sort_values(by=time_col).reset_index(drop=True)


    def load_events_from_csv(self, filepath, time_col='t', space_cols=None):
        """
        Carga eventos desde un archivo CSV.
        """
        df = pd.read_csv(filepath)
        self.load_events_from_df(df, time_col, space_cols)


    def create_grid(self, cell_size):
        """
        Crea una grilla con celdas cuadradas (o cúbicas) de lado fijo `cell_size`.

        Parámetros:
        - cell_size: tamaño del lado de las celdas en cada eje (float)
        """

        assert self.events is not None, "Debes cargar eventos antes de crear la grilla"

        coords = self.events.iloc[:, 1:].values
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        sizes = maxs - mins

        num_cells = np.ceil(sizes / cell_size).astype(int)
        edges = [
            np.linspace(mins[i], mins[i] + num_cells[i] * cell_size, num_cells[i] + 1)
            for i in range(self.dim)
        ]

        self.grid = {
            'cell_size': cell_size,
            'mins': mins,
            'maxs': maxs,
            'num_cells': num_cells,
            'edges': edges
        }


    def locate_event_cell(self, x):
        """
        Dado un vector de coordenadas, devuelve el índice de la celda en la grilla.
        """
        assert self.grid is not None, "La grilla no ha sido creada aún"
        indices = []
        for i in range(self.dim):
            edges = self.grid['edges'][i]
            idx = np.searchsorted(edges, x[i], side='right') - 1
            idx = max(0, min(idx, self.grid['num_cells'][i] - 1))  # Garantiza que esté dentro del rango
            indices.append(idx)
        return tuple(indices)


    def build_sequential_graph(self):
        """
        Construye un grafo secuencial: cada nodo es una celda,
        y las aristas siguen el orden temporal de los eventos.
        """
        coords = self.events.iloc[:, 1:].values
        prev_cell = None
        for x in coords:
            cell = self.locate_event_cell(x)
            if cell not in self.graph:
                # Calcular centro de la celda
                center = []
                for i, idx in enumerate(tqdm(cell)):
                    edge_start = self.grid['edges'][i][idx]
                    edge_end = self.grid['edges'][i][idx + 1]
                    center.append(0.5 * (edge_start + edge_end))
                self.graph.add_node(cell, pos=tuple(center))
            if prev_cell is not None:
                self.graph.add_edge(prev_cell, cell)
            prev_cell = cell

        # --- Eliminar nodos aislados ---
        nodes_to_remove = [n for n in self.graph.nodes if self.graph.degree(n) == 0]
        self.graph.remove_nodes_from(nodes_to_remove)


    def connectivity_count(self, kind='total'):
        """
        Retorna un diccionario que indica cuántos nodos tienen cada grado de conectividad.

        Parámetros:
        - kind: 'total' (grado total), 'in' (grado de entrada), 'out' (grado de salida)

        Retorna:
        - dict: {grado: cantidad de nodos con ese grado}
        """
        if kind == 'total':
            degrees = [deg for _, deg in self.graph.degree()]
        elif kind == 'in':
            assert self.graph.is_directed(), "El grafo no es dirigido"
            degrees = [deg for _, deg in self.graph.in_degree()]
        elif kind == 'out':
            assert self.graph.is_directed(), "El grafo no es dirigido"
            degrees = [deg for _, deg in self.graph.out_degree()]
        else:
            raise ValueError("Parámetro 'kind' debe ser 'total', 'in' o 'out'")


        count = Counter(degrees)
        max_degree = max(count.keys())

        degree_list = list(range(1, max_degree + 1))
        count_list = [count.get(g, 0) for g in degree_list]

        return degree_list, count_list


    def plot_degree_histogram(self, kind='total', xlim=None, ylim=None,
                                dpi=200, log_x=True,**kwargs):
        """
        Muestra un histograma de grados P(k) con escalas logarítmicas.

        Parámetros:
        - kind: 'total', 'in' o 'out' (para grafos dirigidos)
        """
        grados, cuentas = self.connectivity_count(kind)

        # Normalización para obtener P(k)
        total_nodos = sum(cuentas)
        pk = [c / total_nodos for c in cuentas]

        plt.figure(dpi=dpi, figsize=kwargs.pop('figsize', (7,5)))
        plt.plot(grados, pk, marker=kwargs.pop('marker', 'o'),
                            linestyle=kwargs.pop('linestyle', '-'),
                            linewidth=kwargs.pop('linewidth', 1))
        plt.xlabel(r"$k$", fontsize=kwargs.pop('fs_label', 15))
        plt.ylabel(r"$P(k)$", fontsize=kwargs.pop('fs_label', 15))
        if log_x:
            plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.tight_layout()
        plt.show()


    def plot_graph(self, with_labels=False, node_size=100, edge_color='gray',
                    xlabel="x", ylabel="y", **kwargs):
        """
        Dibuja el grafo en 2D, usando las posiciones espaciales de los nodos.

        Parámetros:
        - with_labels: si True, muestra los índices de celda en los nodos.
        - node_size: tamaño de los nodos.
        - edge_color: color de las aristas.
        """
        pos = nx.get_node_attributes(self.graph, 'pos')

        plt.figure(dpi=kwargs.pop('dpi', 200), figsize=kwargs.pop('figsize', (7,7)))
        nx.draw(
            self.graph,
            pos=pos,
            with_labels=with_labels,
            node_size=node_size,
            edge_color=edge_color,
            node_color='skyblue'
        )
        plt.xlabel(fr"${{{xlabel}}}$", fontsize=kwargs.pop('fs_label', 17))
        plt.ylabel(fr"${{{ylabel}}}$", fontsize=kwargs.pop('fs_label', 17))
        plt.axis('equal')
        plt.tight_layout()
        plt.show()



class EvolvingNetwork:
    def __init__(self, point_process_df, time_col='t', space_cols=None, dim=2, directed=False):
        self.df = point_process_df.sort_values(by=time_col).reset_index(drop=True)
        self.time_col = time_col
        self.space_cols = space_cols
        self.dim = dim
        self.directed = directed
        self.snapshots = []  # Lista de tuplas (timestamp, PointProcessNetwork)
        self.stats = []      # Lista de dicts con stats globales por snapshot


    def build_by_time_window(self, window_size, step_size, cell_size=1.0):
        """
        Parámetros:
        - window_size: timedelta o float (segundos)
        - step_size: timedelta o float (segundos)
        """
        t_values = pd.to_datetime(self.df[self.time_col])
        t_start = t_values.min()
        t_end = t_values.max()

        current = t_start
        while current + window_size <= t_end:
            mask = (t_values >= current) & (t_values < current + window_size)
            window_df = self.df[mask]
            if len(window_df) == 0:
                current += step_size
                continue
            timestamp = window_df[self.time_col].mean()

            ppn = PointProcessNetwork(dim=self.dim, directed=self.directed)
            ppn.load_events_from_df(window_df, time_col=self.time_col, space_cols=self.space_cols)
            ppn.create_grid(cell_size)
            ppn.build_sequential_graph()

            self.snapshots.append((timestamp, ppn))
            self.stats.append(self._get_graph_stats(ppn.graph))

            current += step_size


    def build_by_event_window(self, window_size=1000, step=200, cell_size=1.0):
        for start in range(0, len(self.df) - window_size + 1, step):
            end = start + window_size
            window_df = self.df.iloc[start:end]
            timestamp = window_df[self.time_col].mean()

            ppn = PointProcessNetwork(dim=self.dim, directed=self.directed)
            ppn.load_events_from_df(window_df, time_col=self.time_col, space_cols=self.space_cols)
            ppn.create_grid(cell_size)
            ppn.build_sequential_graph()

            self.snapshots.append((timestamp, ppn))
            self.stats.append(self._get_graph_stats(ppn.graph))


    def do_evolution(self, method="event", **kwargs):
        """
        Ejecuta la evolución de la red según el método indicado.

        Parámetros:
        - method: "event" o "time"
        - kwargs: parámetros para pasar al método correspondiente
        """
        if method == "event":
            self.build_by_event_window(**kwargs)
        elif method == "time":
            self.build_by_time_window(**kwargs)
        else:
            raise ValueError("Method must be 'event' or 'time'")


    def _get_graph_stats(self, G):
        return {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G),
            'k_avg': np.mean([d for n, d in G.degree()]) if G.number_of_nodes() > 0 else 0,
        }


    def plot_stat(self, stat_name, **kwargs):
        times = [t for t, _ in self.snapshots]
        values = [s[stat_name] for s in self.stats]
        plt.figure(dpi=200)
        plt.plot(times, values, marker='o')
        plt.title(f"Evolución de {stat_name}")
        plt.xlabel("Tiempo")
        plt.ylabel(stat_name)
        plt.grid(True)
        plt.tight_layout()
        plt.show()



