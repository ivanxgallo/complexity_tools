# Multi Fractality Analysis Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.interpolate import UnivariateSpline, interp1d
from tqdm.notebook import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def _compute_mfdfa_single_q(ts, profile, scales, q_val, order, double):
    N = len(ts)
    flucts = []

    for s in scales:
        num_segments = N // s
        if num_segments < 2:
            continue

        F_nu = []

        # Adelante
        for v in range(num_segments):
            segment = profile[v * s:(v + 1) * s]
            x = np.arange(s)
            poly = np.polyfit(x, segment, order)
            trend = np.polyval(poly, x)
            detrended = segment - trend
            F_nu.append(np.mean(detrended ** 2))

        # Reverso
        if double:
            for v in range(num_segments):
                segment = profile[N - (v + 1) * s:N - v * s]
                x = np.arange(s)
                poly = np.polyfit(x, segment, order)
                trend = np.polyval(poly, x)
                detrended = segment - trend
                F_nu.append(np.mean(detrended ** 2))

        # Función de fluctuación
        if q_val != 0:
            F_s = (np.mean(np.array(F_nu) ** (q_val / 2))) ** (1 / q_val)
        else:
            F_s = np.exp(0.5 * np.mean(np.log(F_nu)))

        flucts.append(F_s)

    return q_val, scales[:len(flucts)], np.array(flucts)


class MultiFractality:
    def __init__(self, time_series):
        """
        Inicializa el objeto con una serie de tiempo unidimensional.
        """
        self.series = np.asarray(time_series)
        self.mfdfa_results = {}
        assert self.series.ndim == 1, "La serie de tiempo debe ser unidimensional"

    def plot_time_series(self, xlabel="Time", ylabel="Value",
                        xlog=False, ylog=False, **kwargs):
        """
        Grafica la serie de tiempo original.

        Parámetros:
        - title: título del gráfico (por defecto "Time Series")
        - xlabel: etiqueta del eje x (por defecto "Time")
        - ylabel: etiqueta del eje y (por defecto "Value")
        - **kwargs: parámetros opcionales para personalizar el gráfico (color, linestyle, marker, etc.)
        """
        plt.figure(dpi=kwargs.pop('dpi', 200), figsize=kwargs.pop('figsize', (10, 5)))
        plt.plot(self.series,
                color=kwargs.pop('color', 'tab:blue'),
                linestyle=kwargs.pop('linestyle', '-'),
                linewidth=kwargs.pop('linewidth', 1.2),
                marker=kwargs.pop('marker', ''),
                alpha=kwargs.pop('alpha', 0.9),
                label=kwargs.pop('label', None),
                **kwargs)
        plt.xlabel(xlabel, fontsize=kwargs.pop('fs_labels', 12))
        plt.ylabel(ylabel, fontsize=kwargs.pop('fs_labels', 12))
        plt.grid(True, linestyle='--', alpha=0.3)

        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')
        if 'label' in kwargs:
            plt.legend(fontsize=kwargs.pop('fs_legend', 10))
        plt.tight_layout()
        plt.show()


    def mfdfa(self, q=2, min_scale=10, max_scale=1000,
                n_scales=50, order=1, double=False, parallel=True):
        """
        Aplica el algoritmo MFDFA a la serie de tiempo para uno o varios valores de q.

        Parámetros:
        - q: valor escalar o lista/array de valores q (puede incluir q=0)
        - min_scale: menor longitud de escala (s)
        - max_scale: mayor longitud de escala (s)
        - n_scales: número de escalas a considerar
        - order: orden del polinomio para el detrending
        - double: si True, usa ventanas al derecho y al revés
        - parallel: si True, calcula para cada q en paralelo

        Almacena:
        - self.mfdfa_results[q]: diccionario con claves "scales" y "flucts" para cada q
        """
        ts = self.series
        N = len(ts)
        mean_ts = np.mean(ts)
        profile = np.cumsum(ts - mean_ts)

        raw_scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num=n_scales)
        scales = np.unique(np.floor(raw_scales).astype(int))

        q_list = [q] if np.isscalar(q) else list(q)

        if parallel:
            with ProcessPoolExecutor() as executor:
                futures = {
                    executor.submit(_compute_mfdfa_single_q, ts, profile, scales, q_val, order, double): q_val
                    for q_val in q_list
                }

                for future in tqdm(as_completed(futures), total=len(futures), desc="MFDFA parallel"):
                    q_val, s_vals, flucts = future.result()
                    self.mfdfa_results[q_val] = {
                        "scales": s_vals,
                        "flucts": flucts
                    }

        else:
            for q_val in tqdm(q_list, desc="MFDFA sequential"):
                q_val, s_vals, flucts = _compute_mfdfa_single_q(ts, profile, scales, q_val, order, double)
                self.mfdfa_results[q_val] = {
                    "scales": s_vals,
                    "flucts": flucts
                }


    def plot_fluctuations(self, q=None, loglog=True, cmap_name='tab10',
                      fs_labels=12, fs_legend=10, **kwargs):
        """
        Grafica F_q(s) vs s para uno o varios valores de q.

        Parámetros:
        - q: valor escalar o lista de valores de q a graficar (si None, se usan todos los disponibles)
        - loglog: si True, usa escala log-log
        - cmap_name: nombre del colormap para distinguir curvas
        - fs_labels: tamaño de fuente de etiquetas
        - fs_legend: tamaño de fuente de leyenda
        - **kwargs: argumentos extra para personalizar cada curva (e.g., alpha, linestyle)
        """
        if not hasattr(self, "mfdfa_results") or len(self.mfdfa_results) == 0:
            raise ValueError("No hay resultados de MFDFA disponibles.")

        # Determinar qué q graficar
        if q is None:
            q_list = sorted(self.mfdfa_results.keys())
        elif isinstance(q, (list, tuple, np.ndarray)):
            q_list = list(q)
        else:
            q_list = [q]

        plt.figure(dpi=kwargs.pop('dpi', 200))
        cmap = get_cmap(cmap_name)

        for i, q_val in enumerate(q_list):
            if q_val not in self.mfdfa_results:
                print(f"[WARN] q = {q_val} no ha sido calculado, se omite.")
                continue

            data = self.mfdfa_results[q_val]
            s = np.array(data["scales"])
            F = np.array(data["flucts"])
            color = cmap(i % cmap.N)

            plt.plot(s, F,
                    label=fr"$q={q_val}$",
                    color=color,
                    alpha=kwargs.get('alpha', 0.8),
                    linestyle=kwargs.get('linestyle', '-'),
                    marker=kwargs.get('marker', 'o'),
                    linewidth=kwargs.get('linewidth', 1.2),
                    markersize=kwargs.get('markersize', 4))

        if loglog:
            plt.xscale('log')
            plt.yscale('log')

        plt.xlabel(r"$s$", fontsize=fs_labels)
        plt.ylabel(r"$F_q(s)$", fontsize=fs_labels)
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.legend(fontsize=fs_legend)
        plt.tight_layout()
        plt.show()



    def fit_hq(self, q=None, s_min=10, s_max=1000):
        """
        Ajusta una recta log-log a F(s) vs s entre s_min y s_max,
        y devuelve la pendiente h(q) para cada q especificado o para todos los q disponibles.

        Parámetros:
        - q: escalar, lista o None. Si es None, se ajusta para todos los q en self.mfdfa_results
        - s_min: escala mínima a considerar
        - s_max: escala máxima a considerar

        Guarda en self.mfdfa_results[q]:
        - h: exponente de Hurst estimado
        - error_h: error estándar del ajuste lineal
        - intercept: A del ajuste log-log
        - range: tupla con (s_min, s_max)
        """
        # Si q es None, usar todas las claves existentes
        if q is None:
            q_list = list(self.mfdfa_results.keys())
        elif not isinstance(q, (list, tuple, np.ndarray)):
            q_list = [q]
        else:
            q_list = q

        for q_val in q_list:
            if q_val not in self.mfdfa_results:
                print(f"[WARN] Debes ejecutar mfdfa para q={q_val} antes de ajustar h(q)")
                continue

            s = np.array(self.mfdfa_results[q_val]["scales"])
            F = np.array(self.mfdfa_results[q_val]["flucts"])

            # Filtrar por el rango dado
            mask = (s >= s_min) & (s <= s_max)
            log_s = np.log(s[mask])
            log_F = np.log(F[mask])

            # Ajuste lineal log-log
            coeffs, cov = np.polyfit(log_s, log_F, 1, cov=True)
            h_q = coeffs[0]
            log_A = coeffs[1]
            A = np.exp(log_A)
            error_h = np.sqrt(cov[0, 0])

            # Guardar resultados
            self.mfdfa_results[q_val].update({
                "h": h_q,
                "error_h": error_h,
                "intercept": A,
                "range": (s_min, s_max)
            })

            print(f"[INFO] Fit for q={q_val}: h(q) = {h_q:.4f} ± {error_h:.4f}")



    def plot_fit_hq(self, q=None, correction=1.0, fs_legend=10, cmap_name='tab20', **kwargs):
        """
        Plotea F_q(s) vs s y su ajuste para uno o varios valores de q ajustados anteriormente.

        Parámetros:
        - q: valor o lista de valores de q ajustados anteriormente (si None, se grafican todos los disponibles con ajuste)
        - correction: factor multiplicativo opcional (se aplica a todos los F_q(s))
        """
        # Determinar lista de q
        if q is None:
            q_list = [k for k in self.mfdfa_results if "h" in self.mfdfa_results[k]]
        elif isinstance(q, (list, tuple, np.ndarray)):
            q_list = list(q)
        else:
            q_list = [q]

        if len(q_list) == 0:
            print("[WARN] No hay valores de q con ajuste realizado (h, intercept).")
            return

        plt.figure(dpi=kwargs.pop('dpi', 200))
        cmap = get_cmap(cmap_name)

        for i, q_val in enumerate(q_list):
            if q_val not in self.mfdfa_results:
                print(f"[WARN] Debes ejecutar mfdfa para q={q_val} antes de graficar.")
                continue
            if "h" not in self.mfdfa_results[q_val] or "intercept" not in self.mfdfa_results[q_val]:
                print(f"[WARN] Falta ajuste para q={q_val}. Ejecuta fit_hq primero.")
                continue

            color = cmap(i % cmap.N)
            s = np.array(self.mfdfa_results[q_val]["scales"])
            F = np.array(self.mfdfa_results[q_val]["flucts"])
            result = self.mfdfa_results[q_val]
            h_q = result["h"]
            A = result["intercept"]
            s_min, s_max = result["range"]

            mask = (s >= s_min) & (s <= s_max)
            s_fit = s[mask]
            F_fit = correction * A * s_fit ** h_q

            # Plot datos y ajuste
            plt.plot(s, correction * F, 'o-', label=fr"Data $q={q_val}$", color=color, alpha=0.5)
            plt.plot(s_fit, F_fit, 'k-', label=fr"$q={q_val}$ fit: $s^{{{h_q:.2f}}}$")

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r"$s$")
        plt.ylabel(r"$F_q(s)$")
        plt.legend(fontsize=fs_legend)
        plt.grid(True, which='both', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()



    def plot_hq_vs_q(self, marker='o', color='tab:blue', fs_labels=12,
                fs_legend=10, interpolate=None, interpolation_marker='-',
                poly_degree=3, num_points=300, ms_interpolate=5,
                spline_smooth=0.5, interp_kind='cubic', **kwargs):
        """
        Grafica h(q) vs q con opción de interpolar la curva.

        Parámetros:
        - marker: marcador para los puntos
        - color: color de la curva de puntos
        - fs_labels: tamaño fuente de etiquetas
        - fs_legend: tamaño fuente de leyenda
        - interpolate: 'polynomial', 'splines', 'interp1d' o None
        - poly_degree: grado del polinomio si interpolate='polynomial'
        - num_points: número de puntos densos para la curva interpolada
        - ms_interpolate: tamaño de marcador para curva interpolada
        - spline_smooth: parámetro de suavizado si interpolate='splines'
        - interp_kind: tipo de interpolación para interp1d ('linear', 'cubic', etc.)
        """
        if not hasattr(self, "mfdfa_results") or len(self.mfdfa_results) == 0:
            raise ValueError("No hay resultados de MFDFA disponibles.")

        # Extraer datos
        qs = sorted(self.mfdfa_results.keys())
        hqs = np.array([self.mfdfa_results[q]['h'] for q in qs])
        errors = np.array([self.mfdfa_results[q]['error_h'] for q in qs])

        # Plot puntos con barras de error
        plt.figure(dpi=kwargs.pop('dpi', 200))
        plt.errorbar(qs, hqs, yerr=errors, fmt=marker,
                    color=color, capsize=3, label=r"$h(q)$")

        # Choosing interpolation method
        if interpolate is not None and len(qs) >= 4:
            q_dense = np.linspace(min(qs), max(qs), num=num_points)

            if interpolate == "polynomial" and len(qs) > poly_degree:
                coeffs = np.polyfit(qs, hqs, deg=poly_degree)
                poly = np.poly1d(coeffs)
                hq_dense = poly(q_dense)
                label_interp = fr"Polynomial deg {poly_degree}"

            elif interpolate == "splines":
                spline = UnivariateSpline(qs, hqs, s=spline_smooth)
                hq_dense = spline(q_dense)
                label_interp = fr"Spline smooth {spline_smooth}"

            elif interpolate == "interp1d":
                interp_func = interp1d(qs, hqs, kind=interp_kind)
                hq_dense = interp_func(q_dense)
                label_interp = fr"interp1d kind '{interp_kind}'"

            else:
                raise ValueError("Interpolación inválida o insuficientes puntos.")

            self.q_dense = q_dense
            self.hq_dense = hq_dense

            # Plot interpolación
            plt.plot(q_dense, hq_dense, interpolation_marker, color='black',
                    alpha=0.7, markersize=ms_interpolate, label=label_interp)

        plt.xlabel(r"$q$", fontsize=fs_labels)
        plt.ylabel(r"$h(q)$", fontsize=fs_labels)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.legend(fontsize=fs_legend)
        plt.show()


    def compute_tau(self, plot=False, **kwargs):
        """
        Calcula la función tau(q) = h(q) * q - 1 para todos los q calculados.

        Parámetros:
        - plot: si True, genera el gráfico de tau(q)
        - **kwargs: parámetros opcionales para personalizar el gráfico

        Retorna:
        - tau: array con tau(q)
        """
        assert hasattr(self, "hq_dense") and hasattr(self, "q_dense"), \
            "Debes ejecutar plot_hq_vs_q antes de calcular tau(q)"

        self.tau = self.hq_dense * self.q_dense - 1

        if plot:
            plt.figure(dpi=kwargs.pop('dpi', 200))  # permite cambiar resolución
            plt.plot(self.q_dense, self.tau,
                    marker=kwargs.pop('marker', 'o'),
                    linestyle=kwargs.pop('linestyle', '-'),
                    color=kwargs.pop('color', 'tab:red'),
                    label=kwargs.pop('label', r"$\tau(q)$"),
                    **kwargs)
            plt.xlabel(r"$q$", fontsize=kwargs.pop('fs_labels', 12))
            plt.ylabel(r"$\tau(q)$", fontsize=kwargs.pop('fs_labels', 12))
            plt.grid(True, which='both', linestyle='--', alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()

        else:
            return self.q_dense, self.tau


    def compute_singularity_spectrum(self, plot=False, **kwargs):
        """
        Calcula el espectro de singularidades f(α) vs α a partir de τ(q).

        Requiere haber ejecutado previamente `plot_hq_vs_q()` y `compute_tau()`.

        Parámetros:
        - plot: si True, grafica f(α) vs α
        - **kwargs: parámetros opcionales para customizar el gráfico

        Retorna:
        - alpha: array con α(q)
        - f_alpha: array con f(α)
        """
        assert hasattr(self, "tau") and hasattr(self, "q_dense"), \
            "Debes ejecutar compute_tau() antes de calcular f(α)"

        # Derivada de τ(q) respecto a q → α(q)
        dq = np.gradient(self.q_dense)
        dtau = np.gradient(self.tau)
        alpha = dtau / dq

        # f(α) = q * α - τ(q)
        f_alpha = self.q_dense * alpha - self.tau

        # Guardar en self
        self.alpha = alpha
        self.f_alpha = f_alpha

        if plot:
            plt.figure(dpi=kwargs.pop('dpi', 200))
            plt.plot(alpha, f_alpha,
                    marker=kwargs.pop('marker', 'o'),
                    linestyle=kwargs.pop('linestyle', '-'),
                    color=kwargs.pop('color', 'tab:purple'),
                    label=kwargs.pop('label', r"$f(\alpha)$"),
                    **kwargs)
            plt.xlabel(r"$\alpha$", fontsize=kwargs.pop('fs_labels', 12))
            plt.ylabel(r"$f(\alpha)$", fontsize=kwargs.pop('fs_labels', 12))
            plt.grid(True, which='both', linestyle='--', alpha=0.3)
            plt.legend(fontsize=kwargs.pop('fs_legend', 10))
            plt.tight_layout()
            plt.show()

        else:
            return alpha, f_alpha


    def compute_Dq(self, method_q1='interp', plot=False, **kwargs):
        """
        Calcula la dimensión generalizada D_q = tau(q) / (q - 1), salvo para q = 1.

        Parámetros:
        - method_q1: cómo tratar el valor en q=1
            - 'interp': interpola usando vecinos
            - 'nan': lo deja como NaN
            - 'omit': lo deja como NaN (puedes filtrarlo después)
        - plot: si True, grafica D_q vs q
        - **kwargs: argumentos opcionales para personalizar el gráfico

        Guarda:
        - self.Dq: array con los valores de D_q
        """
        assert hasattr(self, "tau") and hasattr(self, "q_dense"), \
            "Debes calcular tau(q) primero con compute_tau."

        q = self.q_dense
        tau = self.tau

        Dq = np.full_like(q, np.nan, dtype=np.float64)
        mask = ~np.isclose(q, 1.0)
        Dq[mask] = tau[mask] / (q[mask] - 1)

        if method_q1 == 'interp':
            idx = np.argmin(np.abs(q - 1.0))
            if 0 < idx < len(q) - 1:
                D_left = tau[idx - 1] / (q[idx - 1] - 1)
                D_right = tau[idx + 1] / (q[idx + 1] - 1)
                Dq[idx] = (D_left + D_right) / 2
        elif method_q1 == 'nan':
            pass
        elif method_q1 == 'omit':
            pass

        self.Dq = Dq

        if plot:
            plt.figure(dpi=kwargs.pop('dpi', 200))
            plt.plot(q, Dq, 'o-', color=kwargs.pop('color', 'tab:green'), **kwargs)
            plt.xlabel(r"$q$", fontsize=kwargs.pop('fs_labels', 12))
            plt.ylabel(r"$D_q$", fontsize=kwargs.pop('fs_labels', 12))
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show()

        else:
            return Dq


    def compute_D0(self, method="max"):
        """
        Calcula D_0 con distintos métodos.

        Parámetros:
        - method: "max" usa max(f(α)), "limit" usa lim q→0 de D_q

        Retorna:
        - D_0: valor escalar
        """
        if method == "max":
            assert hasattr(self, "f_alpha"), "Debes calcular f(α) primero con compute_singularity_spectrum()."
            D0 = np.nanmax(self.f_alpha)

        elif method == "limit":
            assert hasattr(self, "Dq"), "Debes calcular D_q primero con compute_Dq()."
            assert hasattr(self, "q_dense"), "Faltan los valores de q para D_q."

            q_vals = self.q_dense
            D_vals = self.Dq

            # Seleccionar valores cercanos a q=0 (ej. -0.5 < q < 0.5, excluyendo q=0)
            mask = (q_vals > -0.5) & (q_vals < 0.5) & (~np.isclose(q_vals, 0))

            if np.sum(mask) < 2:
                raise ValueError("No hay suficientes puntos cercanos a q=0 para calcular el límite.")

            fit = np.polyfit(q_vals[mask], D_vals[mask], deg=1)
            D0 = fit[1]  # Intercepto ≈ D_q(q=0)

        else:
            raise ValueError("Método inválido. Usa 'max' o 'limit'.")

        self.D0 = D0
        return D0


    def remove_q_results(self, q):
        """
        Elimina los resultados almacenados para uno o varios valores específicos de q.

        Parámetro:
        - q: valor escalar o iterable de valores de q que se desean eliminar.

        Efecto:
        - Elimina las entradas correspondientes de self.mfdfa_results
        """
        if not hasattr(self, "mfdfa_results"):
            raise AttributeError("No hay resultados de MFDFA disponibles.")

        # Convertir q a lista si es un valor único
        if isinstance(q, (list, tuple, np.ndarray)):
            q_list = q
        else:
            q_list = [q]

        for q_val in q_list:
            if q_val in self.mfdfa_results:
                del self.mfdfa_results[q_val]
                print(f"[INFO] Resultados para q = {q_val} eliminados.")
            else:
                print(f"[WARN] No se encontraron resultados para q = {q_val}.")



'''
    def mfdfa(self, q=2, min_scale=10, max_scale=1000, n_scales=50, order=1, double=False):
        """
        Aplica el algoritmo MFDFA a la serie de tiempo para uno o varios valores de q.

        Parámetros:
        - q: valor escalar o lista/array de valores q (puede incluir q=0)
        - min_scale: menor longitud de escala (s)
        - max_scale: mayor longitud de escala (s)
        - n_scales: número de escalas a considerar
        - order: orden del polinomio para el detrending
        - double: si True, usa ventanas al derecho y al revés

        Almacena:
        - self.mfdfa_results[q]: diccionario con claves "scales" y "flucts" para cada q
        """
        ts = self.series
        N = len(ts)
        mean_ts = np.mean(ts)
        profile = np.cumsum(ts - mean_ts)

        raw_scales = np.logspace(np.log10(min_scale), np.log10(max_scale), num=n_scales)
        scales = np.unique(np.floor(raw_scales).astype(int))

        # Asegurar formato iterable
        q_list = [q] if np.isscalar(q) else list(q)

        for q_val in tqdm(q_list, desc="Computing MFDFA for each q"):
            flucts = []

            for s in tqdm(scales, desc=f"Scales for q={q_val}", leave=False):
                num_segments = N // s
                if num_segments < 2:
                    continue

                F_nu = []

                # Adelante
                for v in range(num_segments):
                    segment = profile[v * s:(v + 1) * s]
                    x = np.arange(s)
                    poly = np.polyfit(x, segment, order)
                    trend = np.polyval(poly, x)
                    detrended = segment - trend
                    F_nu.append(np.mean(detrended ** 2))

                # Reverso (doble recorrido)
                if double:
                    for v in range(num_segments):
                        segment = profile[N - (v + 1) * s:N - v * s]
                        x = np.arange(s)
                        poly = np.polyfit(x, segment, order)
                        trend = np.polyval(poly, x)
                        detrended = segment - trend
                        F_nu.append(np.mean(detrended ** 2))

                # Función de fluctuación para cada q
                if q_val != 0:
                    F_s = (np.mean(np.array(F_nu) ** (q_val / 2))) ** (1 / q_val)
                else:
                    F_s = np.exp(0.5 * np.mean(np.log(F_nu)))

                flucts.append(F_s)

            # Guardar resultado para este q
            self.mfdfa_results[q_val] = {
                "scales": scales[:len(flucts)],
                "flucts": np.array(flucts)
            }
'''
