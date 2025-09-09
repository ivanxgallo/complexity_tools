import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from tqdm.notebook import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed



def _hvg_visible_edges(i, series):
    """
    Retorna las aristas visibles desde el nodo i hacia j > i.
    """
    N = len(series)
    edges = []
    yi = series[i]
    for j in range(i + 1, N):
        yj = series[j]
        obstructed = False
        for k in range(i + 1, j):
            if series[k] >= min(yi, yj):
                obstructed = True
                break
        if not obstructed:
            edges.append((i, j))
    return edges


def _vg_visible_edges(i, series):
    """
    Retorna las aristas visibles desde el nodo i hacia j > i,
    usando la regla de visibilidad geométrica (VG).
    """
    N = len(series)
    edges = []
    yi = series[i]
    for j in range(i + 1, N):
        yj = series[j]
        visible = True
        for k in range(i + 1, j):
            yk = series[k]
            y_interp = yi + (yj - yi) * (k - i) / (j - i)
            if yk >= y_interp:
                visible = False
                break
        if visible:
            edges.append((i, j))
    return edges


def compare_degree_distributions(vg_list, kind='total', labels=None,
                                colors=None, markers=None, linestyles=None,
                                fit_linestyle='--', fit_color='black',
                                fit_range=None, dpi=200, figsize=(8, 6), **kwargs):
    """
    Compara las distribuciones de conectividad y sus ajustes para múltiples objetos VG.

    Parámetros:
    - vg_list: lista de objetos VisibilityGraph
    - kind: tipo de grado ('total', 'in', 'out')
    - labels: lista opcional de etiquetas para la leyenda
    - colors, markers, linestyles: listas opcionales de estilos
    - fit_linestyle, fit_color: estilo del ajuste
    - fit_range: tupla (k_min, k_max) para forzar el mismo rango de ajuste
    """
    plt.figure(dpi=dpi, figsize=figsize)

    for idx, vg in enumerate(vg_list):
        label = labels[idx] if labels else f"VG {idx+1}"
        color = colors[idx] if colors else None
        marker = markers[idx] if markers else 'o'
        linestyle = linestyles[idx] if linestyles else '-'

        # Recuperar k y pk
        data = vg.degree_distributions.get(kind)
        if data is None:
            print(f"[WARN] VG {idx+1} no tiene distribución almacenada para '{kind}'")
            continue
        k_vals = data["k"]
        pk_vals = data["pk"]

        plt.plot(k_vals, pk_vals, label=label, marker=marker,
                    linestyle=linestyle, color=color)

        # Obtener ajuste (forzar cálculo si hay fit_range)
        if fit_range is not None:
            try:
                vg.fit_degree_slope(kind=kind, k_min=fit_range[0], k_max=fit_range[1],
                                    plot=False, store=True)
            except Exception as e:
                print(f"[ERROR] Ajuste fallido para VG {idx+1}: {e}")
                continue

        fit = vg.degree_fit_results.get(kind)
        if fit:
            slope = fit["slope"]
            intercept = fit["intercept"]
            k_min, k_max = fit["range"]
            log_x = fit["log_x"]
            log_y = fit["log_y"]

            k_fit = k_vals[(k_vals >= k_min) & (k_vals <= k_max)]
            if log_x and log_y:
                pk_fit = np.exp(intercept) * k_fit ** slope
                fit_label = fr"Fit {label}: $P(k) \sim k^{{{slope:.2f}}}$"
                plt.xscale('log')
            elif not log_x and log_y:
                pk_fit = np.exp(intercept + slope * k_fit)
                fit_label = fr"Fit {label}: $P(k) \sim \exp({slope:.2f}k)$"
            else:
                print(f"[WARN] VG {idx+1}: combinación no implementada (log_x={log_x}, log_y={log_y})")
                continue

            plt.plot(k_fit, pk_fit, linestyle=fit_linestyle, color=fit_color, label=fit_label)

    plt.xlabel(r"$k$", fontsize=kwargs.pop("fs_labels", 15))
    plt.ylabel(r"$P(k)$", fontsize=kwargs.pop("fs_labels", 15))
    plt.yscale('log')
    plt.xticks(fontsize=kwargs.pop('xticksize', None))
    plt.yticks(fontsize=kwargs.pop('yticksize', None))
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()


class VisibilityGraph:
    def __init__(self, time_series, kind='VG', directed=None):
        """
        Inicializa el objeto Grafo de Visibilidad.

        Parámetros:
        - time_series: serie de tiempo.
        - kind: 'VG' o 'HVG'.
        - directed: None, 'forward', 'backward' o 'both'.
        """
        self.time_series = np.asarray(time_series)
        self.kind = kind.upper()
        self.directed = directed.lower() if directed is not None else None

        if self.kind not in ['VG', 'HVG']:
            raise ValueError("kind must be either 'VG' or 'HVG'")
        if self.directed not in [None, 'forward', 'backward', 'both']:
            raise ValueError("directed must be one of: None, 'forward', 'backward', 'both'")

        self.graph = nx.Graph() if self.directed is None else nx.DiGraph()


    def build_graph(self, max_workers=None):
        if self.kind == 'VG':
            self._build_vg(max_workers=max_workers)
        elif self.kind == 'HVG':
            self._build_hvg(max_workers=max_workers)


    def _add_edge(self, i, j):
        if self.directed is None:
            self.graph.add_edge(i, j)
        elif self.directed == 'forward' and i < j:
            self.graph.add_edge(i, j)
        elif self.directed == 'backward' and i > j:
            self.graph.add_edge(i, j)
        elif self.directed == 'both':
            self.graph.add_edge(i, j)
            self.graph.add_edge(j, i)


    def _build_hvg(self, max_workers=None, chunksize=10):
        """
        Construye el HVG usando multiprocessing via concurrent.futures,
        respetando la lógica de dirección con self._add_edge.
        """
        series = self.time_series
        N = len(series)

        # Inicializar grafo (según directed)
        self.graph = nx.Graph() if self.directed is None else nx.DiGraph()
        self.graph.add_nodes_from(range(N))

        all_edges = []

        # Paralelizar por nodo i
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_hvg_visible_edges, i, series): i for i in range(N)}

            for future in tqdm(as_completed(futures), total=N, desc="Computing HVG"):
                result = future.result()
                all_edges.extend(result)

        # Aplicar lógica de dirección
        for i, j in all_edges:
            self._add_edge(i, j)


    def _build_vg(self, max_workers=None, chunksize=10):
        """
        Construye el VG usando multiprocessing con concurrent.futures,
        respetando la lógica de dirección con self._add_edge.
        """
        series = self.time_series
        N = len(series)

        # Inicializar grafo
        self.graph = nx.Graph() if self.directed is None else nx.DiGraph()
        self.graph.add_nodes_from(range(N))

        all_edges = []

        # Paralelizar por nodo i
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_vg_visible_edges, i, series): i for i in range(N)}

            for future in tqdm(as_completed(futures), total=N, desc="Computing VG"):
                result = future.result()
                all_edges.extend(result)

        # Aplicar lógica de dirección
        for i, j in all_edges:
            self._add_edge(i, j)


    def connectivity_count(self, kind='total'):
        """
        Retorna un diccionario con la distribución de conectividad P(k),
        es decir, cuántos nodos tienen cada valor de grado.

        Parámetros:
        - kind: 'total' (grado total), 'in' (grado de entrada), 'out' (grado de salida)

        Retorna:
        - degree_list: lista de valores de k
        - count_list: lista con la cantidad de nodos con ese grado
        """
        if self.graph is None:
            raise ValueError("El grafo aún no ha sido construido. Usa build_graph().")

        if kind == 'total':
            degrees = [deg for _, deg in self.graph.degree()]
        elif kind == 'in':
            if not self.graph.is_directed():
                raise ValueError("El grafo no es dirigido. 'in' solo aplica a grafos dirigidos.")
            degrees = [deg for _, deg in self.graph.in_degree()]
        elif kind == 'out':
            if not self.graph.is_directed():
                raise ValueError("El grafo no es dirigido. 'out' solo aplica a grafos dirigidos.")
            degrees = [deg for _, deg in self.graph.out_degree()]
        else:
            raise ValueError("El parámetro 'kind' debe ser 'total', 'in' o 'out'.")

        count = Counter(degrees)
        if not count:
            return [], []

        max_degree = max(count.keys())
        degree_list = list(range(1, max_degree + 1))
        count_list = [count.get(k, 0) for k in degree_list]

        if not hasattr(self, "degree_distributions"):
            self.degree_distributions = {}

        total = sum(count_list)
        k_vals = np.array(degree_list)
        pk_vals = np.array([c / total for c in count_list])

        # Eliminar valores donde P(k) == 0 por robustez
        nonzero_mask = pk_vals > 0
        k_vals = k_vals[nonzero_mask]
        pk_vals = pk_vals[nonzero_mask]

        self.degree_distributions[kind] = {
            "k": k_vals,
            "pk": pk_vals
        }

        return degree_list, count_list


    def plot_degree_histogram(self, kind='total',
                            xlim=None, ylim=None, dpi=200,
                            show_fit=False, correction=1.0,
                            log_x=True, log_y=True, **kwargs):
        """
        Grafica el histograma de conectividad P(k) vs. k con opción de mostrar ajuste.

        Parámetros:
        - kind: 'total', 'in' o 'out'
        - xlim, ylim: límites opcionales de ejes
        - dpi: resolución del gráfico
        - show_fit: si True, se grafica el ajuste si fue previamente almacenado
        - correction: factor multiplicativo aplicado al ajuste (solo visual)
        - log_x, log_y: bools, determinan la escala de cada eje
        - warn_if_fit_missing: si True, emite advertencia si no hay ajuste guardado
        """
        grados, cuentas = self.connectivity_count(kind)

        if not grados:
            print("[WARN] No hay datos de conectividad para graficar.")
            return

        grados = np.array(grados)
        cuentas = np.array(cuentas)
        total_nodos = np.sum(cuentas)
        pk = cuentas / total_nodos

        # Filtrar valores donde pk > 0
        mask_nonzero = pk > 0
        grados = grados[mask_nonzero]
        pk = pk[mask_nonzero]

        plt.figure(dpi=dpi, figsize=kwargs.pop('figsize', (7,5)))
        plt.plot(grados, pk, marker=kwargs.pop('marker','o'),
                    linestyle=kwargs.pop('linestyle','-'),
                    linewidth=kwargs.pop('linewidth',1.5),
                    label=r"$P(k)$ data", **kwargs)
        plt.xlabel(r"$k$", fontsize=kwargs.pop("fs_labels", 15))
        plt.ylabel(r"$P(k)$", fontsize=kwargs.pop("fs_labels", 15))
        plt.xticks(fontsize=kwargs.pop('xticksize', None))
        plt.yticks(fontsize=kwargs.pop('yticksize', None))

        # Mostrar ajuste si está disponible
        if show_fit:
            if hasattr(self, "degree_fit_results") and kind in self.degree_fit_results:
                fit_info = self.degree_fit_results[kind]
                slope = fit_info["slope"]
                intercept = fit_info["intercept"]
                k_min, k_max = fit_info["range"]
                fit_log_x = fit_info.get("log_x", True)
                fit_log_y = fit_info.get("log_y", True)

                k_fit = grados[(grados >= k_min) & (grados <= k_max)]

                # Ajuste según forma guardada
                if fit_log_x and fit_log_y:
                    pk_fit = np.exp(intercept) * k_fit ** slope
                    label = fr"Fit: $P(k) \sim k^{{{slope:.2f}}}$"
                    plt.plot(k_fit, correction * pk_fit, 'k--', linewidth=1.5, label=label)
                elif not fit_log_x and fit_log_y:
                    pk_fit = np.exp(intercept + slope * k_fit)
                    label = fr"Fit: $\log P(k) \sim k$"
                    plt.plot(k_fit, correction * pk_fit, 'k--', linewidth=1.5, label=label)
                else:
                    raise ValueError("Ajuste no implementado para esta combinación de log_x y log_y.")

            else:
                print(f"[WARN] No se encontró ajuste guardado para kind = '{kind}'")

        # Escala de ejes (controlada por argumentos)
        if log_x:
            plt.xscale('log')
        if log_y:
            plt.yscale('log')

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        plt.grid(True, which='both', linestyle='--', alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.show()


    def fit_degree_slope(self, kind='total', k_min=10, k_max=100,
                        plot=True, store=True, correction=1.1,
                        log_x=True, log_y=True, title=True, savepath=None, **kwargs):
        """
        Ajusta una recta a la distribución P(k) vs k transformando opcionalmente los ejes.

        Parámetros:
        - kind: 'total', 'in' o 'out'
        - k_min, k_max: rango de grados k a considerar en el ajuste
        - plot: si es True, se grafica el ajuste junto a los datos
        - store: si es True, guarda el resultado
        - correction: factor multiplicativo para el fit (solo visual)
        - log_x, log_y: si True, se toma logaritmo de ese eje antes del ajuste

        Retorna:
        - pendiente (float), error (float)
        """
        k_vals, p_vals = self.connectivity_count(kind)

        if not k_vals:
            raise ValueError("No hay datos de conectividad disponibles para ajustar.")

        # Arrays y normalización
        k_vals = np.array(k_vals)
        p_vals = np.array(p_vals)
        pk = p_vals / p_vals.sum()

        # Filtrar valores donde pk > 0
        mask_nonzero = pk > 0
        k_vals = k_vals[mask_nonzero]
        pk = pk[mask_nonzero]

        if k_max is None:
            k_max = max(k_vals)

        mask = (k_vals >= k_min) & (k_vals <= k_max)
        if np.sum(mask) < 2:
            raise ValueError("No hay suficientes puntos en el rango especificado.")

        x = k_vals[mask]
        y = pk[mask]

        if log_x:
            x = np.log(x)
        if log_y:
            y = np.log(y)

        coeffs, cov = np.polyfit(x, y, 1, cov=True)
        slope = coeffs[0]
        intercept = coeffs[1]
        error = np.sqrt(cov[0, 0])

        if store:
            if not hasattr(self, "degree_fit_results"):
                self.degree_fit_results = {}
            self.degree_fit_results[kind] = {
                "slope": slope,
                "intercept": intercept,
                "error": error,
                "range": (k_min, k_max),
                "log_x": log_x,
                "log_y": log_y
            }

        if plot:
            plt.figure(dpi=200, figsize=kwargs.pop('figsize', (7,5)))
            plt.plot(k_vals, pk, label=r'$P(k)$ data',
                    marker=kwargs.pop('marker', 'o'),
                    linestyle=kwargs.pop('linestyle','-'),
                    color=kwargs.pop('color', 'red'),
                    linewidth=kwargs.pop('linewidth',1.5), **kwargs)

            k_fit = k_vals[mask]
            if log_x and log_y:
                pk_fit = np.exp(intercept) * k_fit ** slope
                fit_label = fr"$P(k) \sim k^{{{slope:.2f}}}$"
            elif not log_x and log_y:
                pk_fit = np.exp(intercept + slope * k_fit)
                fit_label = fr"$P(k) \sim \exp({slope:.2f}\,k)$"
            else:
                raise ValueError("Ajuste no implementado para esta combinación de log_x y log_y.")

            plt.plot(k_fit, correction * pk_fit, 'k--', label=fit_label)

            if log_x:
                plt.xscale('log')
            if log_y:
                plt.yscale('log')


            if title:
                plt.title(f"Ajuste para '{kind}' (log_x={log_x}, log_y={log_y})")
            plt.xlabel(r"$k$", fontsize=kwargs.pop("fs_labels", 15))
            plt.ylabel(r"$P(k)$", fontsize=kwargs.pop("fs_labels", 15))
            plt.xticks(fontsize=kwargs.pop('xticksize', None))
            plt.yticks(fontsize=kwargs.pop('yticksize', None))
            plt.legend()
            plt.grid(True, which='both', linestyle='--', alpha=0.4)
            plt.tight_layout()
            if savepath:
                plt.savefig(savepath, dpi=kwargs.pop('dpi', 200))
            plt.show()

        print(f"[INFO] Ajuste para '{kind}': pendiente = {slope:.4f} ± {error:.4f}")
        return slope, error



    def plot_time_series(self, xlabel="Time", ylabel="Value",
                        title="Time Series", xlog=False, ylog=False,
                        xlim=None, ylim=None, savepath=None, **kwargs):
        """
        Grafica la serie de tiempo original entregada a la clase.

        Parámetros:
        - xlabel: etiqueta del eje x (por defecto "Time")
        - ylabel: etiqueta del eje y (por defecto "Value")
        - title: título del gráfico (por defecto "Time Series")
        - xlog: bool, escala logarítmica en eje x (False por defecto)
        - ylog: bool, escala logarítmica en eje y (False por defecto)
        - **kwargs: parámetros adicionales para personalizar el gráfico (color, linestyle, marker, etc.)
        """
        if not hasattr(self, 'time_series') or self.time_series is None:
            raise ValueError("No time series data found.")

        x_vals = np.arange(len(self.time_series))

        plt.figure(dpi=kwargs.pop('dpi', 200), figsize=kwargs.pop('figsize', (10, 4)))
        plt.plot(x_vals,
                self.time_series,
                color=kwargs.pop('color', 'tab:blue'),
                linestyle=kwargs.pop('linestyle', '-'),
                linewidth=kwargs.pop('linewidth', 1.5),
                marker=kwargs.pop('marker', ''),
                alpha=kwargs.pop('alpha', 0.9),
                label=kwargs.pop('label', None))

        fs_labels = kwargs.pop('fs_labels', 15)
        plt.ylabel(ylabel, fontsize=fs_labels)
        plt.xlabel(xlabel, fontsize=fs_labels)

        plt.title(title, fontsize=kwargs.pop('fs_title', 14))
        plt.grid(True, linestyle='--', alpha=0.3)

        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')
        if 'label' in kwargs:
            plt.legend(fontsize=kwargs.pop('fs_legend', 10))

        if xlim is not None:
            plt.xlim(xlim)

        if ylim is not None:
            plt.ylim(ylim)

        plt.xticks(fontsize=kwargs.pop('xticksize', None))
        plt.yticks(fontsize=kwargs.pop('yticksize', None))

        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=kwargs.pop('dpi', 200))
        plt.show()