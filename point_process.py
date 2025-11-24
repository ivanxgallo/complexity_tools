# point_process.py
"""
Module for constructing and analyzing point process networks (PPN) from spatiotemporal event data.
Author: Iván Gallo-Méndez
Degree: PhD in Physics
Research Area: Complex Systems and Network Science
Email: ivan.gallo@ug.uchile.cl
Year: 2025
Country: Chile
Institution: University of Chile
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from tqdm.auto import tqdm

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.feature import NaturalEarthFeature
    _HAS_CARTOPY = True
except Exception:
    _HAS_CARTOPY = False


# -------------------------------------- AUXILIAR FUNCTIONS ------------------------------------------ #
# Internal function to generate seasonal intervals according to hemisphere
def get_season_intervals(start, end, hemisphere, season_colors=None):
    intervals = []
    year = start.year
    while year <= end.year:
        if hemisphere.lower() == "south":
            seasons = {
                "Summer": (pd.Timestamp(f"{year}-12-21"), pd.Timestamp(f"{year+1}-03-20")),
                "Autumn": (pd.Timestamp(f"{year}-03-21"), pd.Timestamp(f"{year}-06-20")),
                "Winter": (pd.Timestamp(f"{year}-06-21"), pd.Timestamp(f"{year}-09-22")),
                "Spring": (pd.Timestamp(f"{year}-09-23"), pd.Timestamp(f"{year}-12-20")),
            }
        else:  # hemisferio norte
            seasons = {
                "Winter": (pd.Timestamp(f"{year}-12-21"), pd.Timestamp(f"{year+1}-03-20")),
                "Spring": (pd.Timestamp(f"{year}-03-21"), pd.Timestamp(f"{year}-06-20")),
                "Summer": (pd.Timestamp(f"{year}-06-21"), pd.Timestamp(f"{year}-09-22")),
                "Autumn": (pd.Timestamp(f"{year}-09-23"), pd.Timestamp(f"{year}-12-20")),
            }

        for season, (start_s, end_s) in seasons.items():
            if end_s >= start and start_s <= end:
                intervals.append((start_s, end_s, season_colors[season]))
        year += 1
    return intervals

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
        show_edges=task["show_edges"],
        road_color= task["road_color"],
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


# ----------------------------------------- POINT PROCESS NETWORK ------------------------------------------ #

class PointProcessNetwork:
    def __init__(self, dim=2, directed=False, weighted=False, df=None,
                    time_col=None, space_cols=None, geographic=False):
        if space_cols:
            if isinstance(space_cols, str):
                space_cols = [space_cols]
            assert len(space_cols) == dim, f"Expected {dim} spatial columns, but got {len(space_cols)}."
        self.dim = dim
        self.events = None
        self.grid = None
        self.weighted = weighted
        self.directed = directed
        self.geographic = geographic
        self.graph = nx.DiGraph() if directed else nx.Graph()
        if df is not None and space_cols is not None:
            # Si no se especifica columna de tiempo, se crea un contador
            if time_col is None:
                df = df.copy()
                df.insert(0, 't_counter', range(1, len(df) + 1))
                time_col = 't_counter'
                print(f"[INFO] Time column is not specified. Using '{time_col}' as time index.")
            self.load_events_from_df(df, time_col, space_cols)


    def load_events_from_df(self, df, time_col='t', space_cols=None):
        """
        Carga eventos desde un DataFrame.
        """
        if space_cols is None:
            space_cols = [col for col in df.columns if col != time_col]
        assert len(space_cols) == self.dim, f"Expected {self.dim} spatial columns, but got {len(space_cols)}."
        self.events = df[[time_col] + space_cols].sort_values(by=time_col).reset_index(drop=True)
        self.time_col = time_col
        self.space_cols = space_cols


    def load_events_from_csv(self, filepath, time_col='t', space_cols=None, sep=" ", n_data=None):
        """
        Load events from a CSV file
        """
        if time_col and space_cols:
            df = pd.read_csv(filepath, sep=sep)
            if n_data is None:
                self.load_events_from_df(df, time_col, space_cols)
            else:
                self.load_events_from_df(df.head(n_data), time_col, space_cols)
            self.time_col = time_col
            self.space_cols = space_cols
        else:
            raise ValueError("Must be specified 'time_col' and 'space_cols' to load events from CSV file.")


    def create_grid(self, cell_size):
        """
        Create a spatial grid with (approximately) square or equal-area cells.

        Parameters
        ----------
        cell_size : float
            - If geographic=False: cell side length in coordinate units (e.g., degrees or meters).
            - If geographic=True:  desired cell side length in kilometers (area ≈ cell_size² km²).
        geographic : bool, optional
            If True, adjusts longitudinal spacing with latitude to preserve area on Earth's surface.
        """
        assert self.events is not None, "Must load events before creating the grid"

        coords = self.events.iloc[:, 1:].values
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        sizes = maxs - mins

        if self.geographic and self.dim == 2:
            # --- Conversiones de distancia (km) a grados ---
            mean_lat_deg = np.mean(coords[:, 1])
            mean_lat_rad = np.deg2rad(mean_lat_deg)
            cos_lat = np.cos(mean_lat_rad)

            km_per_deg_lat = 111.32                   # km por grado de latitud
            km_per_deg_lon = 111.32 * cos_lat         # km por grado de longitud (ajustado)

            # Convertir tamaño de celda (en km) a grados
            dlat = cell_size / km_per_deg_lat
            dlon = cell_size / km_per_deg_lon

            # --- Construcción de la grilla en grados ---
            num_cells = np.array([
                np.ceil(sizes[0] / dlon).astype(int),
                np.ceil(sizes[1] / dlat).astype(int)
            ])

            edges = [
                np.linspace(mins[0], mins[0] + num_cells[0] * dlon, num_cells[0] + 1),
                np.linspace(mins[1], mins[1] + num_cells[1] * dlat, num_cells[1] + 1)
            ]

            approx_area_km2 = (cell_size ** 2)

        else:
            # --- Grilla regular (no geográfica) ---
            num_cells = np.ceil(sizes / cell_size).astype(int)
            edges = [
                np.linspace(mins[i], mins[i] + num_cells[i] * cell_size, num_cells[i] + 1)
                for i in range(self.dim)
            ]
            approx_area_km2 = None  # no aplica

        # --- Guardar grilla ---
        self.grid = {
            'cell_size': cell_size,
            'geographic': self.geographic,
            'mins': mins,
            'maxs': maxs,
            'num_cells': num_cells,
            'edges': edges,
            'approx_area_km2': approx_area_km2
        }

        # --- Mensaje informativo ---
        if self.geographic and self.dim == 2:
            print(f"[INFO] Created equal-area grid: ~{approx_area_km2:.2f} km² per cell (Δlat={dlat:.4f}°, Δlon={dlon:.4f}° at {mean_lat_deg:.2f}°)")
        #else:
            #print(f"[INFO] Created regular grid with {num_cells} cells (cell_size={cell_size})")


    def locate_event_cell(self, x):
        """
        Dado un vector de coordenadas, devuelve el índice de la celda en la grilla.
        """
        assert self.grid is not None, "The grid is not created yet."
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
                if self.weighted:
                    # Modo ponderado: acumula frecuencia
                    if self.graph.has_edge(prev_cell, cell):
                        self.graph[prev_cell][cell]["weight"] += 1
                    else:
                        self.graph.add_edge(prev_cell, cell, weight=1)
                else:
                    # Modo simple: solo crea el enlace si no existe
                    self.graph.add_edge(prev_cell, cell)
            prev_cell = cell

        # --- Eliminar nodos aislados ---
        nodes_to_remove = [n for n in self.graph.nodes if self.graph.degree(n) == 0]
        self.graph.remove_nodes_from(nodes_to_remove)

        if self.weighted:
            weights = nx.get_edge_attributes(self.graph, "weight")
            max_w = max(weights.values())
            for u, v in self.graph.edges():
                self.graph[u][v]["weight"] /= max_w


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
            raise ValueError("Parameter 'kind' must be 'total', 'in' or 'out'")

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

        # --- Coeficiente de asortatividad ---
        try:
            assortativity = nx.degree_assortativity_coefficient(self.graph)
            stats['assortativity'] = assortativity
        except Exception as e:
            stats['assortativity'] = None
            print(f"[WARN] Error computing assortativity: {e}")

        # --- numero de comunidades (Louvain) ---
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(self.graph.to_undirected())
            n_communities = len(set(partition.values()))
            stats['n_communities'] = n_communities
        except Exception as e:
            stats['n_communities'] = None
            print(f"[WARN] Error computing number of communities: {e}")

        return stats


    def plot_on_map(self, ax=None, *,
                fs_ticks=15,
                node_attribute="degree",      # NUEVO: atributo a graficar
                percentile=None,              # NUEVO: filtro por percentil
                base_size=2.0,
                max_size=1000,
                edge_width=0.3,
                edge_alpha=0.25,
                node_edgecolor='black',
                node_facecolor='blue',
                road_color='red',
                contour_width=0.4,
                add_background=False,
                add_roads=True,
                add_features=True,
                road_width=1.0,
                extent='auto',
                extent_margin=(1.0, 0.25),
                cities=None,
                city_size=50,
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
            raise ValueError("There are no positions 'pos' in nodes. Excecute build_sequential_graph()")

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


        # roads feature
        if add_roads:
            roads = NaturalEarthFeature(
                category="cultural",  # Tipo de datos: culturales
                name="roads",         # Carreteras
                scale="10m",          # Resolución: 10m es la más detallada
                facecolor="none"      # Sin color de relleno
            )

            ax.add_feature(roads, edgecolor=road_color, linewidth=road_width, alpha=0.7)

        if add_background:
            ax.stock_img()

        # Features base
        if add_features:
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

            gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False

            # Ajustar tamaño de fuente de los ticks (Cartopy)
            gl.xlabel_style = {"size": fs_ticks}
            gl.ylabel_style = {"size": fs_ticks}

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

        # --- Escalado adaptativo del tamaño de los nodos ---
        values = np.array([attr_values[n] for n in nodes_to_plot])
        vmin, vmax = np.min(values), np.max(values)

        # Evitar división por cero si todos son iguales
        if vmax == vmin:
            norm_sizes = np.ones_like(values)
        else:
            norm_sizes = (values - vmin) / (vmax - vmin)

        # Escala final (visual, independiente de los valores absolutos)
        min_size, max_size = base_size, max_size  # puedes ajustar este rango
        scaled_sizes = min_size + norm_sizes * (max_size - min_size)

        # --- Dibujar nodos escalados ---
        for n, size in zip(nodes_to_plot, scaled_sizes):
            lon, lat = pos[n]
            ax.scatter(lon, lat,
                        transform=ccrs.PlateCarree(),
                        s=size,
                        color=node_facecolor,
                        edgecolor=node_edgecolor,
                        linewidths=contour_width,
                        zorder=int(size))

        # Ciudades opcionales
        if cities:
            bigger = max(scaled_sizes)
            for city in cities:
                ax.scatter(city["lon"], city["lat"],
                            s=city_size,
                            color=city_color,
                            edgecolor="black",
                            transform=ccrs.PlateCarree(),
                            zorder=bigger + 1)
                tx_lon, tx_lat = city["text_position"]
                ax.text(tx_lon, tx_lat, city["name"],
                        fontsize=city_textsize, fontweight='bold', color="black",
                        ha="right", va="center",
                        transform=ccrs.PlateCarree(),
                        zorder=bigger + 2)

        if savepath:
            plt.tight_layout()
            plt.savefig(savepath, bbox_inches="tight", dpi=dpi)

        if created_fig and show:
            plt.show()

        return ax


    def compute_advanced(self, what="bc", *, attribute_name=None,
                        backend="networkit", normalized=True,
                        approx=False, approx_eps=0.01, approx_delta=0.1,
                        n_threads=None, get_stats=False):
        """
        Calcula atributos 'avanzados' (rápidos con NetworKit) y los escribe en self.graph.
        Por ahora soporta:
        - what="bc"  -> betweenness centrality

        Parámetros
        ----------
        what : str
            Nombre del cómputo. Por ahora: "bc".
        attribute_name : str | None
            Nombre del atributo a escribir en los nodos. Por defecto:
            - "bc" para betweenness
        backend : {"networkit","networkx"}
            Preferencia de backend. Si networkit no está disponible o falla,
            cae a networkx automáticamente.
        normalized : bool
            Si normalizar el resultado (aplica a bc).
        approx : bool
            Si True (y backend = networkit), usa estimación (más rápida).
        approx_eps : float
            Epsilon para estimación (NetworKit).
        approx_delta : float
            Delta para estimación (NetworKit).
        n_threads : int | None
            Número de hilos para NetworKit. Si None, usa el default.

        Retorna
        -------
        dict
            {nodo: valor} con el atributo calculado.
        """
        G = self.graph
        if G.number_of_nodes() == 0:
            return {}

        # Nombre de atributo por defecto
        if attribute_name is None:
            attribute_name = {"bc": "bc"}.get(what, what)

        # --- Betweenness centrality ---
        if what == "bc":
            try:
                # Caso trivial: sin aristas
                if G.number_of_edges() == 0:
                    zeros = {n: 0.0 for n in G.nodes()}
                    nx.set_node_attributes(G, zeros, attribute_name)
                    return zeros

                # Intentar NetworKit si está disponible y pedido
                if backend == "networkit":
                    try:
                        import networkit as nk
                        if n_threads is not None and n_threads > 0:
                            nk.setNumberOfThreads(int(n_threads))

                        # NetworkX -> NetworKit (índices 0..N-1)
                        nodes = list(G.nodes())
                        idx_of = {n: i for i, n in enumerate(nodes)}
                        nkG = nk.Graph(n=G.number_of_nodes(), weighted=False, directed=G.is_directed())

                        for u, v in G.edges():
                            nkG.addEdge(idx_of[u], idx_of[v])

                        # Betweenness exacta o estimada
                        if approx:
                            algo = nk.centrality.EstimateBetweenness(
                                nkG, epsilon=approx_eps, delta=approx_delta, normalized=normalized
                            )
                        else:
                            algo = nk.centrality.Betweenness(nkG, normalized=normalized)

                        algo.run()
                        scores = algo.scores()  # lista alineada con índices
                        result = {nodes[i]: float(scores[i]) for i in range(len(nodes))}

                        nx.set_node_attributes(G, result, attribute_name)
                        if get_stats:
                            return result

                    except Exception as e:
                        # Fallback a NetworkX si NetworKit no está o falló
                        # (podrías loguear e si quieres más detalle)
                        print(f"[WARN] NetworKit failed: {e}")
                        backend = "networkx"

                # Fallback / opción explícita: NetworkX
                if backend == "networkx":
                    result = nx.betweenness_centrality(G, normalized=normalized)
                    nx.set_node_attributes(G, result, attribute_name)
                    if get_stats:
                        return result
            except Exception as e:
                # Si llega aquí, algo raro pasó en el enrutamiento de backend
                raise print("Could not compute 'bc' with the backends available. ERROR:", e)

        else:
            # Punto de extensión: agrega más elif aquí para otras métricas
            raise ValueError(f"'{what}' is not implemented yet. Use 'bc' for now.")


    def node_attribute_timeseries(self, attribute="degree", *,
                                degree_kind="total",    # 'total' | 'in' | 'out' (solo si attribute='degree')
                                default=np.nan,
                                show_progress=False):
        """
        Construye una serie de tiempo tomando, para cada evento, el valor de un atributo del
        nodo (celda) en el que cae ese evento.

        Parámetros
        ----------
        attribute : str
            Nombre del atributo de nodo a leer. Caso especial: 'degree'.
        degree_kind : {'total','in','out'}
            Si attribute='degree', qué grado reportar (para grafos dirigidos).
        default : float
            Valor a usar si el nodo no existe en self.graph o no tiene el atributo.
        show_progress : bool
            Si True, muestra barra de progreso (tqdm).

        Retorna
        -------
        pd.Series
            Serie indexada por tiempo del evento, con el valor del atributo.
        """
        # Requisitos
        assert self.events is not None, "Primero carga eventos con load_events_from_df()."
        assert self.grid is not None, "Primero crea la grilla con create_grid()."
        assert self.graph is not None, "Primero construye el grafo con build_sequential_graph()."

        # Tiempos y coords
        times = self.events.iloc[:, 0].values
        coords = self.events.iloc[:, 1:].values

        # Diccionarios de lookup de atributo
        if attribute == "degree":
            if degree_kind == "total":
                attr_dict = dict(self.graph.degree())
            elif degree_kind == "in":
                if not self.graph.is_directed():
                    raise ValueError("degree_kind='in' requiere grafo dirigido.")
                attr_dict = dict(self.graph.in_degree())
            elif degree_kind == "out":
                if not self.graph.is_directed():
                    raise ValueError("degree_kind='out' requiere grafo dirigido.")
                attr_dict = dict(self.graph.out_degree())
            else:
                raise ValueError("degree_kind debe ser 'total', 'in' o 'out'.")
            degree_default = 0  # si el nodo no está, consideramos grado 0
        else:
            attr_dict = nx.get_node_attributes(self.graph, attribute)
            degree_default = None  # no aplica

        # Iterar eventos
        iterator = coords
        if show_progress:
            iterator = tqdm(coords, desc="Building attribute time series", leave=False)

        values = []
        for x in iterator:
            cell = self.locate_event_cell(x)  # tuple índice de celda
            if attribute == "degree":
                val = attr_dict.get(cell, degree_default)
            else:
                val = attr_dict.get(cell, default)
            # si sigue faltando (p.ej. degree_default=None en atributos no-degree)
            if val is None:
                val = default
            values.append(val)

        # Serie con índice temporal
        return pd.Series(values, index=pd.to_datetime(times), name=attribute)


    def average_neighbor_degree_by_k(self):
        """
        Calcula k_nn(k): el promedio del grado de los vecinos de los nodos de grado k.

        Guarda internamente:
        - self.k_asort : lista de grados distintos
        - self.knn     : lista del promedio del grado medio de los vecinos para cada k

        Retorna
        -------
        tuple : (k_values, knn_values)
            k_values: lista de grados distintos
            knn_values: promedio del grado medio de los vecinos para cada k
        """
        if self.graph.number_of_nodes() == 0:
            raise ValueError("Graph is empty. Build it first.")

        # --- Grados de nodos ---
        degrees = dict(self.graph.degree())

        # --- Promedio del grado de los vecinos ---
        knn_per_node = {}
        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            k_i = degrees[node]
            if k_i > 0 and len(neighbors) > 0:
                neighbor_degrees = [degrees[nbr] for nbr in neighbors]
                knn_per_node[node] = np.mean(neighbor_degrees)
            else:
                knn_per_node[node] = np.nan  # nodos aislados o sin vecinos

        # --- Agrupar por grado ---
        knn_by_k = {}
        for node, k_i in degrees.items():
            k_nn_i = knn_per_node[node]
            if np.isnan(k_nn_i):
                continue
            if k_i not in knn_by_k:
                knn_by_k[k_i] = []
            knn_by_k[k_i].append(k_nn_i)

        self.k_asort = sorted(knn_by_k.keys())
        self.knn = [np.mean(knn_by_k[k]) for k in self.k_asort]

        return self.k_asort, self.knn


    def plot_assortativity(self, loglog=True, xlim=None, ylim=None,
                           dpi=200, figsize=(7,5), **kwargs):
        """
        Grafica la relación k_nn(k) vs k para evaluar la asortatividad estructural local.

        Usa los valores calculados en average_neighbor_degree_by_k().
        """
        if not hasattr(self, "k_asort") or not hasattr(self, "knn"):
            self.average_neighbor_degree_by_k()

        plt.figure(dpi=dpi, figsize=figsize)
        plt.plot(self.k_asort, self.knn, 'o-',
                    color=kwargs.pop('color', 'steelblue'),
                    label=r"$k_{nn}(k)$")

        if loglog:
            plt.xscale('log')
            plt.yscale('log')

        plt.xlabel(r"$k$", fontsize=kwargs.pop('fs_label', 15))
        plt.ylabel(r"$\langle k_{nn}(k) \rangle$", fontsize=kwargs.pop('fs_label', 15))
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.legend()
        if xlim: plt.xlim(xlim)
        if ylim: plt.ylim(ylim)
        plt.show()


    def global_assortativity(self):
        """
        Calcula la asortatividad global r (coeficiente de Pearson entre grados de nodos conectados).

        Guarda internamente:
        - self.r_global : float, coeficiente de asortatividad global.

        Retorna
        -------
        float : valor del coeficiente global de asortatividad.
        """
        if self.graph.number_of_nodes() == 0:
            raise ValueError("Graph is empty. Build it first.")

        try:
            self.r_global = nx.degree_assortativity_coefficient(self.graph)
        except Exception as e:
            self.r_global = np.nan
            print(f"[WARN] Could not compute global assortativity: {e}")

        return self.r_global


    def detect_communities(self, resolution=1.0, random_state=None):
        """
        Detecta comunidades en el grafo usando el método de Louvain.

        Parámetros
        ----------
        resolution : float, opcional (default=1.0)
            Controla la granularidad de las comunidades detectadas.
            - Valores menores a 1 → menos comunidades (más grandes).
            - Valores mayores a 1 → más comunidades (más pequeñas).
        random_state : int | None, opcional
            Semilla del generador aleatorio para obtener resultados reproducibles.

        Guarda internamente
        -------------------
        - self.communities : dict {nodo: comunidad_id_ordenado}
        - self.community_sizes : dict {comunidad_id_ordenado: número de nodos}
        - self.communities_sorted : lista [(comunidad_id_ordenado, tamaño), ...] ordenadas por tamaño descendente.

        Retorna
        -------
        dict
            Diccionario {nodo: comunidad_id_ordenado}
        """
        if self.graph.number_of_nodes() == 0:
            raise ValueError("Graph is empty. Build it first.")

        try:
            import community as community_louvain  # paquete 'python-louvain'
        except ImportError:
            raise ImportError("Must install the library 'python-louvain': pip install python-louvain")

        # --- Detección de comunidades con control de resolución ---
        if self.weighted:
            partition = community_louvain.best_partition(
                self.graph,
                resolution=resolution,
                random_state=random_state,
                weight = 'weight'
            )
        else:
            partition = community_louvain.best_partition(
                self.graph,
                resolution=resolution,
                random_state=random_state
            )

        # --- Calcular tamaños originales ---
        sizes = Counter(partition.values())

        # --- Ordenar comunidades por tamaño (descendente) ---
        sorted_by_size = sorted(sizes.items(), key=lambda x: x[1], reverse=True)

        # --- Crear mapeo antiguo → nuevo ID ---
        remap = {old_id: new_id+1 for new_id, (old_id, _) in enumerate(sorted_by_size)}

        # --- Reasignar IDs en el diccionario de comunidades ---
        partition_renamed = {node: remap[old_cid] for node, old_cid in partition.items()}

        # --- Guardar estructuras internas ---
        self.communities = partition_renamed
        self.community_sizes = {remap[old_id]: size for old_id, size in sizes.items()}
        self.communities_sorted = sorted(self.community_sizes.items(), key=lambda x: x[1], reverse=True)


        # mensaje informativo
        print(f"[INFO] {len(self.communities_sorted)} detected communities (res={resolution})")

        return self.communities


    def get_community_summary(self, top=None, resolution=1.0, random_state=None):
        """
        Retorna el resumen de comunidades detectadas.

        Parámetros
        ----------
        top : int | None
            Si se indica, retorna solo las N comunidades más grandes.

        Retorna
        -------
        dict : {comunidad_id: tamaño}
        """

        _ = self.detect_communities(resolution=resolution, random_state=random_state)

        if top is None:
            return dict(self.communities_sorted)

        return dict(self.communities_sorted[:top])


    def plot_communities_on_map(self, ax=None, *,
                                fs_ticks=18,
                                fs_legend=15,
                                legend_location='best',
                                communities=None,
                                node_attribute="degree",
                                percentile=None,
                                percentile_mode="global",
                                base_size=2.0,
                                max_size=50,
                                edge_width=0.3,
                                edge_alpha=0.25,
                                node_edgecolor='black',
                                contour_width=0.4,
                                add_features=True,
                                add_roads=True,
                                extent='auto',
                                extent_margin=(1.0, 0.25),
                                cities=None,
                                city_size=70,
                                city_color="gold",
                                city_textsize=12,
                                show_edges=True,
                                base_size_legend=50,
                                savepath=None,
                                dpi=200, figsize=(6, 6),
                                show=True,
                                color_palette=None):
        """
        Dibuja las comunidades sobre un mapa (lon, lat), coloreando por comunidad.

        Parámetros
        ----------
        communities : list[int] | None
            IDs de comunidades a mostrar. Si None, se muestran todas.
        node_attribute : str
            Atributo de los nodos que define el tamaño ('degree', 'bc', etc.)
        """
        if not _HAS_CARTOPY:
            raise ImportError("Cartopy is not installed. Use `pip install cartopy`")

        if not hasattr(self, "communities_sorted"):
            raise ValueError("Must excecute detect_communities() first.")

        # if communities is int then make it a list
        if communities is None:
            raise ValueError("Parameter 'communities' must be a list of community IDs.")
        elif isinstance(communities, int):
            communities = [communities]

        # Crear figura/ax si no viene uno
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                                    dpi=dpi, figsize=figsize)
            created_fig = True

        # Posiciones de nodos
        pos = nx.get_node_attributes(self.graph, 'pos')
        if not pos:
            raise ValueError("No hay posiciones 'pos' en los nodos. Llame a build_sequential_graph() antes.")

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

        if add_roads:
            roads = NaturalEarthFeature(
                category="cultural", name="roads", scale="10m", facecolor="none"
            )
            ax.add_feature(roads, edgecolor='red', linewidth=1, alpha=0.7)

        # Features base
        if add_features:
            ax.add_feature(cfeature.COASTLINE, linewidth=1)
            ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

            gl = ax.gridlines(draw_labels=True, color="gray", alpha=0.5, linestyle="--")
            gl.top_labels = False
            gl.right_labels = False

            # Ajustar tamaño de fuente de los ticks (Cartopy)
            gl.xlabel_style = {"size": fs_ticks}
            gl.ylabel_style = {"size": fs_ticks}

        # Valores de atributo (para tamaño)
        if node_attribute == "degree":
            attr_values = dict(self.graph.degree())
        else:
            sample_node = next(iter(self.graph.nodes))
            if node_attribute not in self.graph.nodes[sample_node]:
                raise ValueError(f"The attribute '{node_attribute}' is not calculated for nodes in graph.")
            attr_values = nx.get_node_attributes(self.graph, node_attribute)

        # --- Filtro por percentil (adaptativo: global o local) ---
        nodes_to_plot = list(self.graph.nodes)  # default
        if percentile is not None:
            if percentile_mode not in ("global", "local"):
                raise ValueError("percentile_mode must be 'global' or 'local'")

            if percentile_mode == "global":
                # --- Global: se calcula un único umbral para toda la red ---
                threshold = np.percentile(list(attr_values.values()), 100 - percentile)
                nodes_to_plot = [n for n, v in attr_values.items() if v >= threshold]

            elif percentile_mode == "local":
                # --- Local: un umbral distinto por comunidad ---
                nodes_to_plot = []
                communities_in_plot = (
                    communities if communities is not None
                    else sorted(set(self.communities.values()))
                )
                for cid in communities_in_plot:
                    members = [n for n, c in self.communities.items() if c == cid]
                    values = [attr_values[n] for n in members if n in attr_values]
                    if len(values) == 0:
                        continue
                    threshold = np.percentile(values, 100 - percentile)
                    selected = [n for n in members if attr_values.get(n, 0) >= threshold]
                    nodes_to_plot.extend(selected)

        # Filtrar comunidades si se pide
        if communities is not None:
            nodes_to_plot = [n for n in nodes_to_plot
                                if self.communities.get(n) in communities]

        # Mostrar aristas
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

        # Paleta de colores para comunidades (manual, más legible)
        community_ids = sorted(set(self.communities[n] for n in nodes_to_plot))

        # Lista con los 10 colores más reconocibles (puedes ajustar tonos)
        if color_palette is None:
            color_palette = [
                "#e41a1c",  # rojo
                "#377eb8",  # azul
                "#4daf4a",  # verde
                "#ff7f00",  # naranja
                "#ffff33",  # amarillo
                "#984ea3",  # violeta
                "#a65628",  # marrón
                "#f781bf",  # rosa
                "#999999",  # gris
                "#66c2a5"   # turquesa
            ]

        # Si hay más comunidades que colores, cicla la lista
        color_map = {cid: color_palette[(cid-1) % len(color_palette)] for cid in community_ids}


        # --- Escalado adaptativo del tamaño de los nodos ---
        values = np.array([attr_values[n] for n in nodes_to_plot])
        vmin, vmax = np.min(values), np.max(values)

        # Evita división por cero si todos los valores son iguales
        if vmax == vmin:
            norm_sizes = np.ones_like(values)
        else:
            norm_sizes = (values - vmin) / (vmax - vmin)

        # Escala final: mapea al rango [base_size, max_size]
        min_size, max_size_plot = base_size, max_size
        scaled_sizes = min_size + norm_sizes * (max_size_plot - min_size)

        # --- Dibujar nodos por comunidad con tamaños escalados ---
        for cid in community_ids:
            members = [n for n in nodes_to_plot if self.communities[n] == cid]
            color = color_map[cid]
            for n in members:
                lon, lat = pos[n]
                # Toma el tamaño escalado correspondiente al nodo
                size = scaled_sizes[nodes_to_plot.index(n)]
                ax.scatter(lon, lat,
                            transform=ccrs.PlateCarree(),
                            s=size,
                            color=color,
                            edgecolor=node_edgecolor,
                            linewidths=contour_width,
                            zorder=int(size),
                            )


        # Ciudades opcionales
        if cities:
            for city in cities:
                ax.scatter(city["lon"], city["lat"],
                            s=city_size, color=city_color,
                            edgecolor="black",
                            transform=ccrs.PlateCarree(), zorder=6)
                tx_lon, tx_lat = city["text_position"]
                ax.text(tx_lon, tx_lat, city["name"],
                        fontsize=city_textsize, fontweight='bold', color="black",
                        ha="right", va="center",
                        transform=ccrs.PlateCarree(), zorder=10)


        # --- LEYENDA PERSONALIZADA ---
        import matplotlib.lines as mlines

        # Ordenar comunidades por tamaño (descendente)
        ordered_communities = sorted(
            [(cid, self.community_sizes.get(cid, 0)) for cid in community_ids],
            key=lambda x: x[1],
            reverse=True
        )

        # Crear handles manuales (círculos)
        handles = []
        for cid, size in ordered_communities:
            color = color_map[cid]
            label = fr"Community {cid} ($n={{{size}}}$)"
            # Marker de igual tamaño visual que los nodos
            marker_size = base_size_legend  # puedes ajustar este factor
            handle = mlines.Line2D([], [], color=color, marker='o', linestyle='None',
                                    markersize=np.sqrt(marker_size), markeredgecolor=node_edgecolor)
            handles.append((cid, handle, label))

        # agregar cities a la leyenda si las hay
        if cities:
            city_handle = mlines.Line2D([], [], color=city_color, marker='o', linestyle='None',
                                        markersize=np.sqrt(base_size_legend), markeredgecolor='black')
            handles.append((
                'city', city_handle, "City"
            ))

        # Crear la leyenda ordenada
        handles_sorted = [h[1] for h in handles]
        labels_sorted = [h[2] for h in handles]
        legend = ax.legend(handles_sorted, labels_sorted,
                            scatterpoints=1,
                            loc=legend_location,
                            fontsize=fs_legend,
                            frameon=True,
                            title=None)
        legend.get_title().set_fontsize(10)

        if savepath:
            plt.tight_layout()
            plt.savefig(savepath, bbox_inches="tight", dpi=dpi)

        if created_fig and show:
            plt.show()

        return ax


    def plot_attribute_distribution(self, attribute, *,
                                    bins=30,
                                    normalize=True,
                                    logbins=True,
                                    pl_fit=False,
                                    fit_bins_range=None,
                                    fit_range=None,
                                    correction = 1.0,
                                    xlog=True,
                                    ylog=True,
                                    xlabel=None,
                                    ylabel=None,
                                    title=None,
                                    figsize=(6,4),
                                    dpi=200,
                                    fs_labels=15,
                                    color="blue",
                                    marker="o-",
                                    alpha=0.8,
                                    linewidth=2,
                                    savepath=None,
                                    show=True):

        # --- Obtener valores ---
        if attribute == "degree":
            values = [d for _, d in self.graph.degree()]
        elif attribute == "bc":
            sample = next(iter(self.graph.nodes))
            if "bc" not in self.graph.nodes[sample]:
                raise ValueError("The attribute 'bc' is not calculated.")
            values = list(nx.get_node_attributes(self.graph, "bc").values())
        else:
            sample = next(iter(self.graph.nodes))
            if attribute not in self.graph.nodes[sample]:
                raise ValueError(f"The attribute '{attribute}' is not present.")
            values = list(nx.get_node_attributes(self.graph, attribute).values())

        if len(values) == 0:
            raise ValueError("No values to plot.")

        # Etiquetas
        if xlabel is None:
            xlabel = attribute
        if ylabel is None:
            ylabel = "PDF" if normalize else "Counts"

        plt.figure(figsize=figsize, dpi=dpi)

        # --- LOGARITHMIC BINS ---
        if logbins:
            vals_pos = np.array([v for v in values if v > 0])
            min_val, max_val = vals_pos.min(), vals_pos.max()
            bins = np.logspace(np.log10(min_val), np.log10(max_val), bins)

        # --- Histograma ---
        h, b = np.histogram(values, bins=bins, density=True if normalize else False)

        # Filtrar ceros
        b_plot = np.array(b[:-1])
        h_plot = np.array(h)
        mask = h_plot > 0
        b_plot = b_plot[mask]
        h_plot = h_plot[mask]

        # --- Plot principal ---
        plt.plot(b_plot, h_plot, marker, color=color, alpha=alpha,
                linewidth=linewidth, label=attribute)


        # --- Fit power law con error ---
        if pl_fit:
            from scipy.stats import linregress

            # Selección de puntos del fit
            mask_fit = (b_plot > 0) & (h_plot > 0)

            if fit_bins_range is not None:
                xmin, xmax = fit_bins_range
                mask_fit &= (b_plot >= xmin) & (b_plot <= xmax)

            x_fit = b_plot[mask_fit]
            y_fit = h_plot[mask_fit]

            if len(x_fit) >= 2:

                # Fit log-log con error estándar
                lr = linregress(np.log(x_fit), np.log(y_fit))
                a = lr.slope                     # exponente
                c = lr.intercept                 # prefactor
                a_err = lr.stderr                # error del exponente
                c_err = lr.intercept_stderr      # error del coeficiente

                # Rango para graficar la línea
                if fit_range is None:
                    xmin, xmax = fit_bins_range
                else:
                    xmin, xmax = fit_range

                x_line = np.logspace(np.log10(xmin), np.log10(xmax), 200)
                y_line = correction * np.exp(c) * x_line**a   # power law line

                # Plot del fit
                plt.plot(x_line, y_line, "k-", linewidth=2,
                        label=fr"Fit $\sim x^{{{a:.2f} \pm {a_err:.2f}}}$")

        # Escalas
        if xlog:
            plt.xscale("log")
        if ylog:
            plt.yscale("log")

        plt.xlabel(xlabel, fontsize=fs_labels)
        plt.ylabel(ylabel, fontsize=fs_labels)
        if title:
            plt.title(title, fontsize=fs_labels)

        plt.grid(True, alpha=0.3)
        if pl_fit:
            plt.legend()

        if savepath:
            plt.savefig(savepath, bbox_inches="tight", dpi=dpi)

        if show:
            plt.show()
        else:
            return b_plot, h_plot





# ----------------------------------------- Evolving Network --------------------------------------------- #

class EvolvingNetwork:
    def __init__(self, point_process_df, time_col='t', space_cols=None,
                    dim=2, directed=False, timestamp_method='mean', save_graphs=False):
        # --- Normalización de space_cols ---
        if space_cols is None:
            # si no te pasan space_cols, tomamos todas menos la de tiempo (si existe)
            if time_col is not None and time_col in point_process_df.columns:
                space_cols = [c for c in point_process_df.columns if c != time_col]
            else:
                # si tampoco hay time_col, toma todas (ya filtraremos al armar cols)
                space_cols = list(point_process_df.columns)
        elif isinstance(space_cols, str):
            space_cols = [space_cols]

        assert isinstance(space_cols, (list, tuple)), "'space_cols' must be string, list or tuple"
        assert all(isinstance(col, str) for col in space_cols), "All elements of 'space_cols' must be strings"
        assert len(space_cols) == dim, f"It is expected {dim} spatial columns, but {len(space_cols)} are given."

        df = point_process_df.copy()

        # --- Manejo de time_col=None -> contador 1..N ---
        if time_col is None:
            self._time_is_counter = True
            time_col = 't_counter'
            # contador desde 1 a N (puedes usar 0..N-1 si prefieres)
            df.insert(0, time_col, np.arange(1, len(df) + 1, dtype=int))
        else:
            self._time_is_counter = False
            # coaccionar a datetime para el modo "time window"
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')

        # --- Selección de columnas (tiempo + espaciales) ---
        cols = [time_col] + list(space_cols)
        df = df[cols].copy()

        # Orden temporal (funciona con datetime o con contador numérico)
        self.df = df.sort_values(by=time_col).reset_index(drop=True)

        # guardar metadatos
        self.time_col = time_col
        self.space_cols = list(space_cols)
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


    def build_by_time_window(self, window_size, step_size, cell_size=0.01, advanced=None):
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

                if advanced:
                    if not isinstance(advanced, list):
                        advanced = [advanced]
                    for ad in advanced:
                        ppn.compute_advanced(what=ad)

                if self.save_graph:
                    self.graphs.append(ppn.graph)

            except Exception as e:
                print(f"[WARN] Window {timestamp} is omitted (error: {type(e).__name__} - {e})")
                continue


    def build_by_event_window(self, window_size=1000, step_size=200, cell_size=0.01, advanced=None):
        for start in tqdm(range(0, len(self.df) - window_size + 1, step_size),
                            desc="Evolving Network by Event Window"):
            end = start + window_size
            window_df = self.df.iloc[start:end]
            timestamp = self._compute_timestamp(window_df[self.time_col])

            ppn = PointProcessNetwork(dim=self.dim, directed=self.directed)
            ppn.load_events_from_df(window_df, time_col=self.time_col, space_cols=self.space_cols)
            ppn.create_grid(cell_size)
            ppn.build_sequential_graph()

            self.time.append(timestamp)
            self.stats.append(ppn.get_graph_stats())

            if advanced:
                if not isinstance(advanced, list):
                    advanced = [advanced]
                for ad in advanced:
                    ppn.compute_advanced(what=ad)

            if self.save_graph:
                self.graphs.append(ppn.graph)


    def do_evolution(self, method="event", **kwargs):
        """
        Ejecuta la evolución de la red según el método indicado.
        """
        self.time = []
        self.stats = []
        self.graphs = []

        if method == "event":
            self.build_by_event_window(**kwargs)
        elif method == "time":
            if getattr(self, "_time_is_counter", False):
                raise ValueError(
                    "The method 'time' needs real timestamps. "
                    "time_col is None. Use method='event'."
                )
            self.build_by_time_window(**kwargs)
        else:
            raise ValueError("Method must be 'event' or 'time'")


    def plot_stat(self, stat_name, figsize=(10, 5), seasons=False,
                hemisphere="south", dpi=200, rotation=45,
                font_size=18, color='black', marker='o',
                savepath=None, ylim=None, show=True, alpha_season=0.7):
        """
        Plotea la evolución temporal de una métrica específica de la red,
        con opción de fondo estacional según el hemisferio.

        Parámetros
        ----------
        stat_name : str
            Nombre de la métrica a graficar (clave de self.stats).
            Ejemplo: 'n_nodes', 'entropy', 'clustering', etc.

        figsize : tuple, opcional
            Tamaño de la figura (default: (10, 5)).

        stations : bool, opcional
            Si True, colorea el fondo según las estaciones del año.

        hemisphere : {'south', 'north'}
            Define el hemisferio para las estaciones (default: 'south').

        dpi : int
            Resolución de la figura.

        rotation : int
            Rotación de las etiquetas del eje X.

        font_size : int
            Tamaño base de fuente.

        color, marker : str
            Color y marcador de la curva principal.

        savepath : str | None
            Si se entrega, guarda la figura en esa ruta.

        ylim : tuple | None
            Límite del eje Y.

        show : bool
            Si False, no muestra el gráfico (útil para generar reportes automáticos).
        """
        import matplotlib.patches as mpatches

        # --- Validaciones básicas ---
        if not hasattr(self, "stats") or len(self.stats) == 0:
            raise ValueError("No statistics registered. Excecute 'do_evolution() first'.")
        if stat_name not in self.stats[0]:
            raise KeyError(f"'{stat_name}' is not a valid metric. Available keys: {list(self.stats[0].keys())}")

        # --- Extraer serie temporal ---
        times = pd.to_datetime(self.time)
        values = [s[stat_name] for s in self.stats]

        # --- Crear figura y eje ---
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # --- Fondo estacional opcional ---
        if seasons:
            season_colors = {
                "Summer": "#FFDD44",
                "Autumn": "#D2B48C",
                "Winter": "#ADD8E6",
                "Spring": "#90EE90",
            }

            start_date, end_date = min(times), max(times)
            intervals = get_season_intervals(start_date, end_date, hemisphere, season_colors)

            # Fondo de color por estación
            for start_s, end_s, color_patch in intervals:
                ax.axvspan(start_s, end_s, color=color_patch, alpha=alpha_season)

            # Leyenda de estaciones
            season_order = ["Summer", "Autumn", "Winter", "Spring"] if hemisphere == "south" \
                            else ["Winter", "Spring", "Summer", "Autumn"]

            legend_patches = [mpatches.Patch(color=season_colors[s], label=s) for s in season_order]
            fig.legend(
                handles=legend_patches,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.08),   # ← sube la leyenda (ajusta 1.05–1.15 según el caso)
                ncol=4,
                fontsize=font_size - 4,
                frameon=False,
                title=None
            )

        # --- Curva principal ---
        ax.plot(times, values, color=color, marker=marker, linestyle='-', linewidth=1.5)
        ax.set_ylabel(stat_name, fontsize=font_size)
        ax.set_xlabel("Time", fontsize=font_size)
        ax.tick_params(axis="x", rotation=rotation, labelsize=font_size - 4)
        ax.tick_params(axis="y", labelsize=font_size - 4)
        ax.grid(True, alpha=0.4)
        if ylim:
            ax.set_ylim(ylim)

        # --- Estilo y guardado ---
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        if savepath:
            plt.savefig(savepath, bbox_inches='tight', dpi=dpi)
        if show:
            plt.show()
        plt.close(fig)



    def plot_all_stats(self, last=None, figsize=(10, 15), seasons=False,
                        hemisphere="south", dpi=200, rotation=45, font_size=18,
                        savepath=None, alpha_season=0.7):
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


        if seasons:
            import matplotlib.patches as mpatches
            season_colors = {
                "Summer": "#FFDD44",  # Amarillo dorado
                "Autumn": "#D2B48C",  # Marrón claro
                "Winter": "#ADD8E6",  # Celeste
                "Spring": "#90EE90",  # Verde claro
            }

            # Obtener rango temporal
            start_date = pd.to_datetime(min(times))
            end_date = pd.to_datetime(max(times))
            intervals = get_season_intervals(start_date, end_date, hemisphere, season_colors)

            # Pintar fondo de estaciones
            for ax in axes:
                for start_s, end_s, color in intervals:
                    ax.axvspan(start_s, end_s, color=color, alpha=alpha_season)

            # Leyenda de estaciones
            legend_patches = [
                mpatches.Patch(color=c, label=s) for s, c in season_colors.items()
            ]
            fig.legend(
                handles=legend_patches,
                loc="upper center",
                ncol=4,
                fontsize=font_size - 3,
                frameon=False,
                title=None
            )

        # Plots individuales
        for ax, (name, values) in zip(axes, metrics.items()):
            ax.plot(times, values, marker='.', linestyle='-', color='black')
            ax.set_ylabel(name, fontsize=font_size)
            ax.grid(True)
            ax.tick_params(axis="y", labelsize=font_size - 4)

        # Formato eje X
        axes[-1].set_xlabel("Time", fontsize=font_size)
        plt.xticks(rotation=rotation, fontsize=font_size - 2)

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        if savepath:
            plt.savefig(savepath, bbox_inches='tight', dpi=dpi)
        plt.show()


    def make_map_gif(self, out_path, *,
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
                        node_facecolor='blue',
                        road_color='red',
                        contour_width=0.4,
                        add_features=True,
                        show=False,
                        loop=0,
                        n_jobs=1,
                        max_side=None):

        import os, tempfile, shutil, traceback
        from PIL import Image
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import imageio.v3 as iio

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
                road_color=road_color,
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


    def event_counts(self, freq='W', *, plot=True,
                    color='red', linewidth=2, xlabel='Date',
                    figsize=(12, 5), dpi=200, savepath=None,
                    show=True, **kwargs):
        """
        Cuenta eventos por ventana temporal usando resample (D, W, M, ...).

        Parámetros
        ----------
        freq : str
            Frecuencia de remuestreo (p.ej. 'D' diario, 'W' semanal, 'M' mensual).
        plot : bool
            Si True, dibuja la serie.
        color, linewidth : estilo del trazo.
        xlabel, ylabel, title : etiquetas del gráfico.
        figsize, dpi : tamaño y resolución de la figura.
        savepath : str | None
            Si se entrega, guarda la figura en esa ruta.
        show : bool
            Si True, muestra la figura.

        Retorna
        -------
        pd.Series
            Serie de conteo de eventos indexada por fecha (periodos).
        """
        # Si el "tiempo" es un contador (time_col=None en el constructor)
        if getattr(self, "_time_is_counter", False):
            raise ValueError(
                "No hay timestamps reales (inicializaste con time_col=None). "
                "event_counts() requiere fechas para re-muestrear por calendario."
            )

        interval_dict = {
            "W": "Week",
            "M": "Month",
            "D": "Day",
            "H": "Hour",
            "T": "Minute",
            "S": "Second"
        }

        tcol = self.time_col
        df = self.df.copy()

        # Asegurar tipo datetime (por si entró como string)
        df[tcol] = pd.to_datetime(df[tcol], errors='coerce')
        if df[tcol].isna().all():
            raise ValueError(f"Column '{tcol}' has no valid datetime values.")

        # Resample (conteo por ventana temporal)
        counts = df.resample(freq, on=tcol).size()
        try:
            counts.name = f'Events_per_{interval_dict[freq]}'
        except KeyError:
            raise ValueError(f"Frequency '{freq}' not recognized. Use one of: {list(interval_dict.keys())}")

        if plot:
            plt.figure(figsize=figsize, dpi=dpi)
            counts.plot(color=color, linewidth=linewidth)
            fs_label = kwargs.pop('fs_labels', 15)
            plt.xlabel(xlabel, fontsize=fs_label)
            plt.ylabel(f"Number of Events per {interval_dict[freq]}", fontsize=fs_label)
            plt.grid(True, alpha=0.4)
            fs_tick = kwargs.pop('fs_ticks', 14)
            plt.xticks(rotation=kwargs.pop('rotation', 45), fontsize=fs_tick)
            plt.yticks(fontsize=fs_tick)
            plt.tight_layout()
            if savepath:
                plt.savefig(savepath, bbox_inches='tight', dpi=dpi)
            if show:
                plt.show()
            else:
                plt.close()
        else:
            return counts







