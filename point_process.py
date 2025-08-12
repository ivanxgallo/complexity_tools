# point_process.py

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import traceback
from collections import Counter
from tqdm.auto import tqdm
import os, tempfile, shutil
import imageio.v2 as imageio
import imageio.v3 as iio
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False


# Worker para un frame (nivel de módulo)
def _render_map_frame(task):
    """
    task: dict con:
    - idx, graph, time_label (str o None), out_path (png)
    - plot kwargs: degree_weight, base_size, size_per_unit, ...
    - extent, extent_margin, cities, dpi, figsize, time_loc, time_fontsize, time_box
    """
    # Reutilizar tu PPN
    ppn = PointProcessNetwork(dim=task.get("dim", 2), directed=task.get("directed", False))
    ppn.graph = task["graph"]

    # Dibujar
    ax = ppn.plot_on_map(
        ax=None,
        #degree_weight=task["degree_weight"],
        base_size=task["base_size"],
        size_per_unit=task["size_per_unit"],
        edge_width=task["edge_width"],
        edge_alpha=task["edge_alpha"],
        node_edgecolor=task["node_edgecolor"],
        node_facecolor=task["node_facecolor"],
        contour_width=task["contour_width"],
        add_features=task["add_features"],
        extent=task["extent"],
        extent_margin=task["extent_margin"],
        cities=task["cities"],
        dpi=task["dpi"],
        figsize=task["figsize"],
        show=task["show"],
        node_attribute=task["node_attribute"],
        percentile=task["percentile"],
        show_edges=task["show_edges"]
    )

    # Timestamp en el frame (opcional)
    label = task.get("time_label", None)
    if label:
        locs = {
            "ul": (0.02, 0.98, "left",  "top"),
            "ur": (0.98, 0.98, "right", "top"),
            "ll": (0.02, 0.02, "left",  "bottom"),
            "lr": (0.98, 0.02, "right", "bottom"),
        }
        x, y, ha, va = locs.get(task["time_loc"], (0.02, 0.98, "left", "top"))
        bbox = dict(facecolor='white', alpha=0.65, edgecolor='none', pad=2) if task["time_box"] else None
        ax.text(x, y, label, transform=ax.transAxes,
                fontsize=task["time_fontsize"], fontweight='bold', ha=ha, va=va, bbox=bbox)

    # Guardar SIEMPRE con la figura del eje
    fig = ax.figure
    fig.savefig(task["out_path"], dpi=task["dpi"], bbox_inches='tight')  # sin bbox_inches='tight' para tamaño consistente
    plt.close(fig)
    return task["out_path"]



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


    def load_events_from_csv(self, filepath, time_col='t', space_cols=None, sep=" ", n_data=None):
        """
        Carga eventos desde un archivo CSV.
        """
        df = pd.read_csv(filepath, sep=sep)
        if n_data is None:
            self.load_events_from_df(df, time_col, space_cols)
        else:
            self.load_events_from_df(df.head(n_data), time_col, space_cols)


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
        for x in tqdm(coords, leave=False, desc="Build a Graph"):
            cell = self.locate_event_cell(x)
            if cell not in self.graph:
                # Calcular centro de la celda
                center = []
                for i, idx in enumerate(cell):
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


    def connectivity_count(self, kind='total', return_prob=False):
        """
        Retorna listas de grados y cuentas, y opcionalmente P(k).

        Parámetros:
        - kind: 'total', 'in', 'out'
        - return_prob: si True, también retorna P(k)

        Retorna:
        - grados, cuentas [, pk]
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

        if return_prob:
            total_nodes = sum(count_list)
            pk_list = [c / total_nodes for c in count_list]
            return degree_list, pk_list

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


    def get_graph_stats(self):
        stats = {
            'n_nodes': self.graph.number_of_nodes(),
            'n_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'k_avg': np.mean([d for _, d in self.graph.degree()]) if self.graph.number_of_nodes() > 0 else 0,
        }

        # --- Clustering global ---
        try:
            if self.graph.is_directed():
                # Para grafos dirigidos se convierte a no dirigido
                G_undirected = self.graph.to_undirected()
                stats['clustering'] = nx.average_clustering(G_undirected)
            else:
                stats['clustering'] = nx.average_clustering(self.graph)
        except Exception as e:
            stats['clustering'] = None
            print(f"[WARN] Error computing clustering: {e}")

        # --- Entropía de conectividad ---
        try:
            _, pk = self.connectivity_count(return_prob=True)
            entropy = -sum(p * np.log(p) for p in pk if p > 0)
            stats['entropy'] = entropy
        except Exception as e:
            stats['entropy'] = None
            print(f"[WARN] Error computing entropy: {e}")

        self.stats = stats

        return stats


    def plot_on_map(self, ax=None, *,
                node_attribute="degree",      # NUEVO: atributo a graficar
                percentile=None,              # NUEVO: filtro por percentil
                base_size=2.0,
                size_per_unit=1.0,             # escalado genérico
                edge_width=0.3,
                edge_alpha=0.25,
                node_edgecolor='black',
                node_facecolor='red',
                contour_width=0.4,
                add_features=True,
                extent='auto',
                extent_margin=(1.0, 0.25),
                cities=None,
                city_size=70,
                city_color="gold",
                city_textsize=12,
                show_edges=True,               # NUEVO: mostrar o no enlaces
                savepath=None,
                dpi=200, figsize=(6, 6),
                show=True):
        """
        Dibuja el grafo sobre un mapa (lon, lat) usando un atributo como tamaño de nodo.
        - node_attribute: "degree" o cualquier otro atributo de nodo ya calculado.
        - percentile: filtra los nodos mostrando solo los que estén en el top X% del atributo.
        - show_edges: si False, no dibuja aristas.
        """
        if not _HAS_CARTOPY:
            raise ImportError("Cartopy is not installed. Use `pip install cartopy`")

        # Crear figura/ax si no viene uno
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                                    dpi=dpi, figsize=figsize)
            created_fig = True

        # Posiciones de nodos
        pos = nx.get_node_attributes(self.graph, 'pos')
        if not pos:
            raise ValueError("No hay posiciones 'pos' en los nodos. ¿Llamaste a build_sequential_graph()?")

        # Extent automático
        if extent == 'auto':
            lons = [p[0] for p in pos.values()]
            lats = [p[1] for p in pos.values()]
            if not lons:
                raise ValueError("No hay nodos para determinar extent.")
            dx, dy = extent_margin
            extent = (min(lons) - dx, max(lons) + dx, min(lats) - dy, max(lats) + dy)

        if extent is not None:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        # Features base
        if add_features:
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
            gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False

        # Obtener valores del atributo
        if node_attribute == "degree":
            attr_values = dict(self.graph.degree())
        else:
            # Verifica si el atributo existe
            sample_node = next(iter(self.graph.nodes))
            if node_attribute not in self.graph.nodes[sample_node]:
                raise ValueError(f"El atributo '{node_attribute}' no está calculado en los nodos.")
            attr_values = nx.get_node_attributes(self.graph, node_attribute)

        # Filtro por percentil
        if percentile is not None:
            threshold = np.percentile(list(attr_values.values()), 100 - percentile)
            nodes_to_plot = [n for n, v in attr_values.items() if v >= threshold]
        else:
            nodes_to_plot = list(self.graph.nodes)

        # Mostrar aristas si corresponde
        if show_edges:
            for u, v in self.graph.edges():
                if u in nodes_to_plot and v in nodes_to_plot:
                    lon_u, lat_u = pos[u]
                    lon_v, lat_v = pos[v]
                    ax.plot([lon_u, lon_v], [lat_u, lat_v],
                            transform=ccrs.PlateCarree(),
                            linewidth=edge_width,
                            color='black',
                            alpha=edge_alpha,
                            zorder=4)

        # Dibujar nodos
        for n in nodes_to_plot:
            lon, lat = pos[n]
            size = base_size + (attr_values[n] * size_per_unit)
            ax.scatter(lon, lat,
                        transform=ccrs.PlateCarree(),
                        s=size,
                        color=node_facecolor,
                        edgecolor=node_edgecolor,
                        linewidths=contour_width,
                        zorder=5)

        # Ciudades opcionales
        if cities:
            for city in cities:
                ax.scatter(city["lon"], city["lat"],
                            s=city_size,
                            color=city_color,
                            edgecolor="black",
                            transform=ccrs.PlateCarree(),
                            zorder=6)
                tx_lon, tx_lat = city["text_position"]
                ax.text(tx_lon, tx_lat, city["name"],
                        fontsize=city_textsize, fontweight='bold', color="black",
                        ha="right", va="center",
                        transform=ccrs.PlateCarree(),
                        zorder=10)

        if savepath:
            plt.tight_layout()
            plt.savefig(savepath, bbox_inches="tight", dpi=dpi)

        if created_fig and show:
            plt.show()

        return ax



class EvolvingNetwork:
    def __init__(self, point_process_df, time_col='t', space_cols=None,
                    dim=2, directed=False, timestamp_method='mean', save_graphs=False):
        # --- Normalización de space_cols ---
        if space_cols is None:
            space_cols = [col for col in point_process_df.columns if col != time_col]
        elif isinstance(space_cols, str):
            space_cols = [space_cols]  # caso dim = 1

        # --- Validación ---
        assert isinstance(space_cols, (list, tuple)), "'space_cols' must be string, list or tuple"
        assert all(isinstance(col, str) for col in space_cols), "All elements of 'space_cols' must bestrings"
        assert len(space_cols) == dim, f"It is expected {dim} spatial columns, but {len(space_cols)} are given."

        # --- Selección de columnas ---
        cols = [time_col] + list(space_cols)
        point_process_df = point_process_df[cols].copy()
        point_process_df[time_col] = pd.to_datetime(point_process_df[time_col])
        self.df = point_process_df.sort_values(by=time_col).reset_index(drop=True)
        self.time_col = time_col
        self.space_cols = space_cols
        self.dim = dim
        self.directed = directed
        self.timestamp_method = timestamp_method
        self.time = []
        self.stats = []
        self.graphs = []
        self.save_graph = save_graphs


    def _compute_timestamp(self, series):
        """
        Retorna un timestamp representativo de la serie según el método definido.

        Parámetros:
        - series: columna de tiempo de una ventana (tipo pd.Series)

        Retorna:
        - Timestamp
        """
        method = self.timestamp_method
        if method == 'mean':
            return series.mean()
        elif method == 'median':
            return series.median()
        elif method == 'min':
            return series.min()
        elif method == 'max':
            return series.max()
        elif method in ('mid', 'center'):
            return series.min() + (series.max() - series.min()) / 2
        else:
            raise ValueError(f"timestamp_method '{method}' no reconocido. Usa: 'mean', 'median', 'min' o 'max'.")


    def build_by_time_window(self, window_size, step_size, cell_size=0.01):
        """
        Divide el dataset en ventanas de tiempo deslizantes, construyendo grafos secuenciales.

        Parámetros:
        - window_size: pd.Timedelta, str o np.timedelta64. Duración de cada ventana.
        - step_size: pd.Timedelta, str o np.timedelta64. Paso con que se desliza la ventana.
        - cell_size: tamaño de la celda para discretizar el espacio.
        """
        # Convertir a pd.Timedelta si vienen como strings o floats
        window_size = pd.to_timedelta(window_size)
        step_size = pd.to_timedelta(step_size)

        t_values = pd.to_datetime(self.df[self.time_col])
        t_start = t_values.min()
        t_end = t_values.max()

        total_steps = int((t_end - t_start - window_size) / step_size) + 1
        current = t_start

        for _ in tqdm(range(total_steps), desc="Evolving Network by Time Window"):
            mask = (t_values >= current) & (t_values < current + window_size)
            window_df = self.df[mask]
            current += step_size  # avanzar paso al principio para no olvidar

            if len(window_df) == 0:
                continue

            try:
                timestamp = self._compute_timestamp(window_df[self.time_col])

                ppn = PointProcessNetwork(dim=self.dim, directed=self.directed)
                ppn.load_events_from_df(window_df, time_col=self.time_col, space_cols=self.space_cols)
                ppn.create_grid(cell_size)
                ppn.build_sequential_graph()

                self.time.append(timestamp)
                self.stats.append(ppn.get_graph_stats())
                if self.save_graph:
                    self.graphs.append(ppn.graph)

            except Exception as e:
                print(f"[WARN] Se omite ventana en {timestamp} (error: {type(e).__name__} - {e})")
                continue


    def build_by_event_window(self, window_size=1000, step_size=200, cell_size=0.01):
        for start in tqdm(range(0, len(self.df) - window_size + 1, step_size), desc="Evolving Network by Event Window"):
            end = start + window_size
            window_df = self.df.iloc[start:end]
            timestamp = self._compute_timestamp(window_df[self.time_col])

            ppn = PointProcessNetwork(dim=self.dim, directed=self.directed)
            ppn.load_events_from_df(window_df, time_col=self.time_col, space_cols=self.space_cols)
            ppn.create_grid(cell_size)
            ppn.build_sequential_graph()

            self.time.append(timestamp)
            self.stats.append(ppn.get_graph_stats())

            if self.save_graph:
                    self.graphs.append(ppn.graph)


    def do_evolution(self, method="event", **kwargs):
        """
        Ejecuta la evolución de la red según el método indicado.

        Parámetros:
        - method: "event" o "time"
        - kwargs: parámetros para pasar al método correspondiente
        """

        self.time = []
        self.stats = []
        self.graphs = []

        if method == "event":
            self.build_by_event_window(**kwargs)
        elif method == "time":
            self.build_by_time_window(**kwargs)
        else:
            raise ValueError("Method must be 'event' or 'time'")


    def plot_stat(self, stat_name, **kwargs):
        times = self.time
        values = [s[stat_name] for s in self.stats]
        plt.figure(dpi=200)
        plt.plot(times, values, marker='o')
        plt.title(f"Evolución de {stat_name}")
        plt.xlabel("Tiempo")
        plt.ylabel(stat_name)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def plot_all_stats(self, last=None, figsize=(10, 15), dpi=200, x_rotation=45, font_size=18):
        """
        Plotea todas las métricas principales de la evolución de la red en subplots separados.

        Parámetros:
        - last: si es un entero negativo, selecciona los últimos N puntos. Si None, usa todo.
        - figsize: tamaño de la figura
        - dpi: resolución del gráfico
        - x_rotation: rotación de etiquetas en el eje X
        - font_size: tamaño base de etiquetas
        """

        # --- Preparar series temporales ---
        if last is not None:
            times = self.time[:last]
            stats = self.stats[:last]
        else:
            times = self.time
            stats = self.stats

        # Extraer series
        n_nodes   = [s['n_nodes'] for s in stats]
        n_edges   = [s['n_edges'] for s in stats]
        density   = [s['density'] for s in stats]
        k_mean    = [s['k_avg'] for s in stats]
        clustering = [s['clustering'] for s in stats]
        entropy   = [s['entropy'] for s in stats]

        # Definir métricas a graficar
        metrics = {
            r"$N$": n_nodes,
            r"$E$": n_edges,
            r"$\rho/\rho_{max}$": np.array(density) / max(density) if max(density) > 0 else density,
            r"$\langle k \rangle$": k_mean,
            r"$C/C_{max}$": np.array(clustering) / max(clustering) if max(clustering) > 0 else clustering,
            r"$S$": entropy,
        }

        # Crear figura
        fig, axes = plt.subplots(nrows=len(metrics), figsize=figsize, sharex=True, dpi=dpi)

        for ax, (name, values) in zip(axes, metrics.items()):
            ax.plot(times, values, marker='.', linestyle='-', color='black')
            ax.set_ylabel(name, fontsize=font_size)
            ax.grid(True)
            ax.tick_params(axis="y", labelsize=font_size - 4)

        # Formato eje X
        axes[-1].set_xlabel("Time", fontsize=font_size)
        plt.xticks(rotation=x_rotation, fontsize=font_size - 2)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.show()


    def make_map_gif(self, out_path, *,
                    fps=4,
                    cities=None,
                    extent="auto_union",      # 'auto' | 'auto_union' | tuple(min_lon,max_lon,min_lat,max_lat) | None
                    extent_margin=(1.0, 0.25),
                    annotate_time=True,
                    time_fmt="%Y-%m-%d",
                    time_loc="ul",            # 'ul','ur','ll','lr'
                    time_fontsize=12,
                    time_box=True,
                    dpi=200, figsize=(6, 6),
                    # Parámetros que se pasan a plot_on_map:
                    degree_weight=True,
                    base_size=2.0,
                    size_per_degree=1.0,
                    edge_width=0.3,
                    edge_alpha=0.25,
                    node_edgecolor='black',
                    node_facecolor='red',
                    contour_width=0.4,
                    add_features=True,
                    show=False,
                    loop=0):
        """
        Genera un GIF con los snapshots guardados en self.graphs dibujados sobre mapa.
        Requiere EvolvingNetwork(..., save_graphs=True) y do_evolution() ya ejecutado.

        - extent:
            'auto_union': usa bounding box global de todos los snapshots (recomendado)
            'auto': cada frame autoajusta su extent (puede "temblar")
            tuple: (min_lon, max_lon, min_lat, max_lat)
            None: no fija extent
        - annotate_time: si True, escribe self.time[idx] en cada frame
        - time_loc: esquina ('ul' arriba-izq, 'ur' arriba-der, 'll' abajo-izq, 'lr' abajo-der)
        """
        if not self.save_graph:
            raise ValueError("No graphs saved. Initialize with save_graphs=True and execute do_evolution().")
        if len(self.graphs) == 0:
            raise ValueError("self.graphs is empty. Did you execute do_evolution()?")

        # --- Calcular extent fijo si se pide 'auto_union' ---
        fixed_extent = None
        if extent == "auto_union":
            all_lons, all_lats = [], []
            for G in self.graphs:
                pos = nx.get_node_attributes(G, 'pos')
                if not pos:
                    continue
                lons = [p[0] for p in pos.values()]
                lats = [p[1] for p in pos.values()]
                all_lons.extend(lons)
                all_lats.extend(lats)
            if len(all_lons) == 0:
                raise ValueError("No hay posiciones 'pos' en los nodos de los snapshots.")
            dx, dy = extent_margin
            fixed_extent = (min(all_lons) - dx, max(all_lons) + dx, min(all_lats) - dy, max(all_lats) + dy)
        elif isinstance(extent, tuple):
            fixed_extent = extent
        elif extent in ("auto", None):
            fixed_extent = extent
        else:
            raise ValueError("Parametro 'extent' inválido. Usa 'auto_union', 'auto', tuple o None.")

        # --- Carpeta temporal para frames ---
        tmpdir = tempfile.mkdtemp(prefix="mapgif_")
        frame_paths = []

        try:
            for idx, G in enumerate(tqdm(self.graphs, desc="Rendering GIF frames")):
                # dummy PPN para reutilizar plot_on_map
                ppn = PointProcessNetwork(dim=self.dim, directed=self.directed)
                ppn.graph = G

                # Dibujar frame
                ax = ppn.plot_on_map(
                    ax=None,
                    degree_weight=degree_weight,
                    base_size=base_size,
                    size_per_degree=size_per_degree,
                    edge_width=edge_width,
                    edge_alpha=edge_alpha,
                    node_edgecolor=node_edgecolor,
                    node_facecolor=node_facecolor,
                    contour_width=contour_width,
                    add_features=add_features,
                    extent=fixed_extent if fixed_extent is not None else 'auto',
                    extent_margin=extent_margin,
                    cities=cities,
                    dpi=dpi, figsize=figsize,
                    show=show
                )

                # --- Timestamp en el frame ---
                if annotate_time and idx < len(self.time):
                    ts = self.time[idx]
                    try:
                        label = ts.strftime(time_fmt)
                    except Exception:
                        label = str(ts)

                    # Coordenadas en el sistema del eje (0..1), no dependen del extent
                    locs = {
                        "ul": (0.02, 0.98, "left",  "top"),
                        "ur": (0.98, 0.98, "right", "top"),
                        "ll": (0.02, 0.02, "left",  "bottom"),
                        "lr": (0.98, 0.02, "right", "bottom"),
                    }
                    x, y, ha, va = locs.get(time_loc, locs["ul"])
                    bbox = dict(facecolor='white', alpha=0.65, edgecolor='none', pad=2) if time_box else None

                    ax.text(x, y, label,
                            transform=ax.transAxes,
                            fontsize=time_fontsize,
                            fontweight='bold',
                            ha=ha, va=va,
                            bbox=bbox)

                # Guardar frame
                frame_path = os.path.join(tmpdir, f"frame_{idx:04d}.png")
                plt.savefig(frame_path, bbox_inches="tight", dpi=dpi)
                plt.close()
                frame_paths.append(frame_path)

            # --- Escribir GIF ---
            images = [imageio.imread(p) for p in frame_paths]
            duration = 1.0 / float(fps)
            imageio.mimsave(out_path, images, duration=duration, loop=loop)

        finally:
            # Limpiar temporales
            shutil.rmtree(tmpdir, ignore_errors=True)

        return out_path


    def make_map_gif_faster(self, out_path, *,
                        duration=1,               # segundos por frame
                        cities=None,
                        extent="auto_union",
                        extent_margin=(1.0, 0.25),
                        annotate_time=True,
                        time_fmt="%Y-%m-%d",
                        time_loc="ul",
                        time_fontsize=12,
                        time_box=True,
                        dpi=200, figsize=(6, 6),
                        # Parámetros de plot_on_map:
                        show_edges=True,
                        node_attribute="degree",
                        percentile=None,
                        #degree_weight=True,
                        base_size=2.0,
                        size_per_unit=1.0,
                        edge_width=0.3,
                        edge_alpha=0.25,
                        node_edgecolor='black',
                        node_facecolor='red',
                        contour_width=0.4,
                        add_features=True,
                        show=False,
                        loop=0,
                        n_jobs=1,
                        max_side=None):

        if not self.save_graph:
            raise ValueError("No graphs saved. Initialize with save_graphs=True and execute do_evolution().")
        if len(self.graphs) == 0:
            raise ValueError("self.graphs is empty. Did you execute do_evolution()?")

        # ---- extent fijo (opcional) ----
        fixed_extent = None
        if extent in ("df", "auto_union"):
            # usar bounds desde el dataframe de origen (más rápido/estable)
            if self.dim < 2 or len(self.space_cols) < 2:
                raise ValueError("It is required at least 2 spatial columns (lon/lat) to set extent from DF.")
            lon_col, lat_col = self.space_cols[0], self.space_cols[1]

            # filtrar NaN/inf
            lons = self.df[lon_col].to_numpy()
            lats = self.df[lat_col].to_numpy()
            mask = np.isfinite(lons) & np.isfinite(lats)
            if not mask.any():
                raise ValueError("No valid spatial values (lon/lat) in the DataFrame to determine extent.")

            lons = lons[mask]; lats = lats[mask]
            dx, dy = extent_margin
            fixed_extent = (float(np.min(lons) - dx),
                            float(np.max(lons) + dx),
                            float(np.min(lats) - dy),
                            float(np.max(lats) + dy))
        elif isinstance(extent, tuple):
            fixed_extent = extent
        elif extent in ("auto", None):
            fixed_extent = extent  # delega el auto por frame (puede 'respirar')
        else:
            raise ValueError("Invalid 'extent'. Use 'df', 'auto_union', 'auto', tuple or None.")

        # ---- carpeta temporal ----
        tmpdir = tempfile.mkdtemp(prefix="mapgif_")
        frame_paths = [os.path.join(tmpdir, f"frame_{i:04d}.png") for i in range(len(self.graphs))]

        # ---- preparar tareas ----
        tasks = []
        for idx, G in enumerate(self.graphs):
            time_label = None
            if annotate_time and idx < len(self.time):
                ts = self.time[idx]
                try:
                    time_label = ts.strftime(time_fmt)
                except Exception:
                    time_label = str(ts)

            tasks.append(dict(
                idx=idx,
                graph=G,
                time_label=time_label,
                out_path=frame_paths[idx],
                #degree_weight=degree_weight,
                base_size=base_size,
                size_per_unit=size_per_unit,
                edge_width=edge_width,
                edge_alpha=edge_alpha,
                node_edgecolor=node_edgecolor,
                node_facecolor=node_facecolor,
                contour_width=contour_width,
                add_features=add_features,
                extent=(fixed_extent if fixed_extent is not None else 'auto'),
                extent_margin=extent_margin,
                cities=cities,
                dpi=dpi,
                figsize=figsize,
                time_loc=time_loc,
                time_fontsize=time_fontsize,
                time_box=time_box,
                dim=self.dim,
                directed=self.directed,
                node_attribute=node_attribute,
                percentile=percentile,
                show_edges=show_edges,
                show=show
            ))

        # ---- render serial o paralelo ----
        try:
            if n_jobs == 1:
                for t in tqdm(tasks, desc="Rendering GIF frames"):
                    try:
                        _render_map_frame(t)
                    except Exception as e:
                        print(f"[ERROR] Frame idx={t['idx']} falló: {type(e).__name__}: {e}")
                        traceback.print_exc()
                        # Si quieres continuar y dejar ese frame en blanco, comenta el raise:
                        raise
            else:
                with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                    futures = [ex.submit(_render_map_frame, t) for t in tasks]
                    for fut in tqdm(as_completed(futures), total=len(futures), desc="Rendering GIF frames"):
                        try:
                            fut.result()
                        except Exception as e:
                            print(f"[ERROR] Frame en proceso paralelo falló: {type(e).__name__}: {e}")
                            traceback.print_exc()
                            raise

            # ---- leer frames, quitar alfa y (opcional) reescalar ----
            frames_np = []
            for j, p in enumerate(frame_paths):
                img = iio.imread(p)
                if img.ndim == 3 and img.shape[2] == 4:
                    img = img[:, :, :3]  # RGB
                if max_side is not None:
                    h, w = img.shape[:2]
                    if max(h, w) > max_side:
                        scale = max_side / max(h, w)
                        new_size = (int(w*scale), int(h*scale))
                        img = np.array(Image.fromarray(img).resize(new_size, Image.LANCZOS))
                frames_np.append(img)

            # ---- escribir GIF con imageio v3 ----
            # duration en milisegundos (int); puede ser lista por-frame si quieres
            dur_ms = int(duration * 1000)
            iio.imwrite(out_path, frames_np, duration=dur_ms, loop=loop)  # loop=0 = infinito

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        return out_path






