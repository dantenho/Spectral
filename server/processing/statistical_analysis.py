"""
Análise Estatística Avançada

Implementa métodos estatísticos sofisticados para análise de dados sensoriais
"""

import numpy as np
from scipy import stats, signal, optimize
from scipy.special import gamma, factorial
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from collections import deque
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# DISTRIBUTION ANALYSIS - Análise de Distribuições
# ============================================================================

@dataclass
class DistributionFit:
    """Resultado de ajuste de distribuição"""
    distribution: str
    parameters: Dict
    ks_statistic: float
    p_value: float
    aic: float  # Akaike Information Criterion
    bic: float  # Bayesian Information Criterion
    log_likelihood: float


class DistributionAnalyzer:
    """
    Analisa e ajusta distribuições estatísticas aos dados

    Testa múltiplas distribuições e retorna a melhor
    """

    DISTRIBUTIONS = [
        'norm',      # Normal (Gaussiana)
        'lognorm',   # Log-normal
        'expon',     # Exponencial
        'gamma',     # Gamma
        'beta',      # Beta
        'weibull_min',  # Weibull
        'rayleigh',  # Rayleigh
        't',         # Student's t
        'chi2',      # Chi-squared
        'laplace'    # Laplace
    ]

    def __init__(self):
        pass

    def fit_best_distribution(self, data: np.ndarray) -> DistributionFit:
        """
        Testa múltiplas distribuições e retorna a melhor

        Usa KS test e critérios de informação (AIC, BIC)
        """
        best_fit = None
        best_aic = np.inf

        for dist_name in self.DISTRIBUTIONS:
            try:
                fit = self._fit_distribution(data, dist_name)

                if fit.aic < best_aic:
                    best_aic = fit.aic
                    best_fit = fit

            except:
                continue

        if best_fit is None:
            # Fallback para normal
            return self._fit_distribution(data, 'norm')

        return best_fit

    def _fit_distribution(
        self,
        data: np.ndarray,
        dist_name: str
    ) -> DistributionFit:
        """Ajusta uma distribuição específica"""

        # Obter distribuição do scipy
        dist = getattr(stats, dist_name)

        # Ajustar parâmetros
        params = dist.fit(data)

        # KS test
        ks_stat, p_value = stats.kstest(data, dist_name, args=params)

        # Log-likelihood
        log_likelihood = np.sum(dist.logpdf(data, *params))

        # AIC = 2k - 2ln(L)
        k = len(params)
        aic = 2 * k - 2 * log_likelihood

        # BIC = k*ln(n) - 2ln(L)
        n = len(data)
        bic = k * np.log(n) - 2 * log_likelihood

        # Parâmetros em dict
        param_names = ['loc', 'scale'] if len(params) == 2 else [f'param{i}' for i in range(len(params))]
        param_dict = {name: float(val) for name, val in zip(param_names, params)}

        return DistributionFit(
            distribution=dist_name,
            parameters=param_dict,
            ks_statistic=float(ks_stat),
            p_value=float(p_value),
            aic=float(aic),
            bic=float(bic),
            log_likelihood=float(log_likelihood)
        )


# ============================================================================
# TIME SERIES ANALYSIS - Análise de Séries Temporais
# ============================================================================

class TimeSeriesAnalyzer:
    """
    Análise estatística de séries temporais

    Implementa testes de estacionariedade, tendências, sazonalidade
    """

    @staticmethod
    def augmented_dickey_fuller_test(data: np.ndarray) -> Dict:
        """
        ADF Test (Augmented Dickey-Fuller) para estacionariedade

        H₀: Série tem raiz unitária (não-estacionária)
        H₁: Série é estacionária

        Equação:
        Δyₜ = α + βt + γyₜ₋₁ + δ₁Δyₜ₋₁ + ... + δₚΔyₜ₋ₚ + εₜ
        """
        # Implementação simplificada do ADF test
        # Scipy não tem ADF nativo, usar regressão linear

        y = data[1:]
        y_lag = data[:-1]
        dy = np.diff(data)

        # Regressão: dy = alpha + beta * y_lag
        X = np.column_stack([np.ones(len(y_lag)), y_lag])

        try:
            beta = np.linalg.lstsq(X, dy, rcond=None)[0]
            residuals = dy - X @ beta

            # Estatística t
            se = np.sqrt(np.sum(residuals**2) / (len(dy) - 2))
            t_stat = beta[1] / (se + 1e-10)

            # P-value aproximado (distribuição t)
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(dy) - 2))

            # Decisão (p < 0.05 = rejeitar H₀ = estacionária)
            is_stationary = p_value < 0.05

        except:
            t_stat = 0
            p_value = 1.0
            is_stationary = False

        return {
            'test_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_stationary': bool(is_stationary)
        }

    @staticmethod
    def detect_trend(data: np.ndarray) -> Dict:
        """
        Detecta tendência usando Mann-Kendall test

        Equação:
        S = Σᵢ Σⱼ₍ⱼ>ᵢ₎ sgn(xⱼ - xᵢ)

        Teste não-paramétrico para tendência monotônica
        """
        n = len(data)

        # Calcular S
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(data[j] - data[i])

        # Variância de S
        var_s = n * (n - 1) * (2 * n + 5) / 18

        # Z-score
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Classificação da tendência
        has_trend = p_value < 0.05

        if has_trend:
            if s > 0:
                trend = "increasing"
            else:
                trend = "decreasing"
        else:
            trend = "no_trend"

        # Magnitude da tendência (Theil-Sen slope)
        slopes = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                slopes.append((data[j] - data[i]) / (j - i))

        slope = np.median(slopes) if slopes else 0

        return {
            'has_trend': bool(has_trend),
            'trend_direction': trend,
            'slope': float(slope),
            'z_statistic': float(z),
            'p_value': float(p_value)
        }

    @staticmethod
    def autocorrelation_analysis(
        data: np.ndarray,
        max_lag: int = 50
    ) -> Dict:
        """
        Análise de autocorrelação (ACF)

        Equação:
        ρₖ = Cov(Yₜ, Yₜ₋ₖ) / Var(Yₜ)
        """
        # Normalizar
        data_centered = data - np.mean(data)
        c0 = np.dot(data_centered, data_centered) / len(data)

        # Calcular ACF
        acf = np.zeros(max_lag + 1)
        acf[0] = 1.0

        for k in range(1, max_lag + 1):
            if k >= len(data):
                break

            ck = np.dot(data_centered[:-k], data_centered[k:]) / len(data)
            acf[k] = ck / c0 if c0 > 0 else 0

        # Encontrar primeiro lag significativo
        # Limiar: 1.96 / sqrt(n) (95% confidence)
        threshold = 1.96 / np.sqrt(len(data))

        significant_lags = np.where(np.abs(acf) > threshold)[0]

        if len(significant_lags) > 1:
            first_significant = significant_lags[1]  # Ignorar lag 0
        else:
            first_significant = 0

        return {
            'acf': acf.tolist(),
            'lags': list(range(max_lag + 1)),
            'first_significant_lag': int(first_significant),
            'max_acf': float(np.max(acf[1:])) if len(acf) > 1 else 0.0
        }


# ============================================================================
# CHANGE POINT DETECTION - Detecção de Pontos de Mudança
# ============================================================================

class ChangePointDetector:
    """
    Detecta pontos de mudança em séries temporais

    Usa algoritmo PELT (Pruned Exact Linear Time)
    """

    @staticmethod
    def detect_change_points(
        data: np.ndarray,
        penalty: float = 10.0,
        min_segment_length: int = 10
    ) -> Dict:
        """
        Detecta pontos de mudança usando custo baseado em variância

        Algoritmo simplificado de change point detection

        Equação de custo:
        C(y₁:ₙ) = Σᵢ (yᵢ - μ)² + β * K

        Onde K = número de change points, β = penalidade
        """
        n = len(data)

        if n < 2 * min_segment_length:
            return {
                'change_points': [],
                'num_segments': 1,
                'costs': []
            }

        # Binary segmentation simplificado
        change_points = []

        def segment_cost(segment):
            """Custo de um segmento (variância)"""
            if len(segment) == 0:
                return 0
            return np.sum((segment - np.mean(segment)) ** 2)

        def find_best_split(segment, start_idx):
            """Encontra melhor ponto de divisão"""
            best_cost = np.inf
            best_split = None

            for i in range(min_segment_length, len(segment) - min_segment_length):
                left = segment[:i]
                right = segment[i:]

                cost = segment_cost(left) + segment_cost(right) + penalty

                if cost < best_cost:
                    best_cost = cost
                    best_split = start_idx + i

            return best_split, best_cost

        # Recursivamente dividir
        def recursive_split(segment, start_idx, depth=0):
            if len(segment) < 2 * min_segment_length or depth > 5:
                return

            split_idx, cost = find_best_split(segment, start_idx)

            if split_idx is not None:
                # Comparar com custo sem split
                no_split_cost = segment_cost(segment)

                if cost < no_split_cost:
                    change_points.append(split_idx)

                    # Recursivamente aplicar às sub-partes
                    recursive_split(segment[:split_idx - start_idx], start_idx, depth + 1)
                    recursive_split(segment[split_idx - start_idx:], split_idx, depth + 1)

        recursive_split(data, 0)

        change_points = sorted(change_points)

        # Calcular custos dos segmentos
        segments = []
        prev_idx = 0

        for cp in change_points + [n]:
            segment = data[prev_idx:cp]
            segments.append(segment_cost(segment))
            prev_idx = cp

        return {
            'change_points': [int(cp) for cp in change_points],
            'num_segments': len(change_points) + 1,
            'segment_costs': [float(c) for c in segments]
        }


# ============================================================================
# EXTREME VALUE ANALYSIS - Análise de Valores Extremos
# ============================================================================

class ExtremeValueAnalyzer:
    """
    Análise de valores extremos (EVA)

    Usa distribuição GEV (Generalized Extreme Value)
    """

    @staticmethod
    def fit_gev(data: np.ndarray) -> Dict:
        """
        Ajusta distribuição GEV (Generalized Extreme Value)

        Equação:
        F(x) = exp(-(1 + ξ(x - μ)/σ)^(-1/ξ))

        Onde:
        - μ = location
        - σ = scale
        - ξ = shape (xi)
        """
        try:
            # Extrair máximos de blocos
            block_size = len(data) // 10
            if block_size < 2:
                block_size = 2

            maxima = []
            for i in range(0, len(data) - block_size + 1, block_size):
                block = data[i:i + block_size]
                maxima.append(np.max(block))

            maxima = np.array(maxima)

            # Ajustar GEV
            shape, loc, scale = stats.genextreme.fit(maxima)

            # Return period (período de retorno)
            # P(X > x) = 1 / return_period

            # Níveis de retorno para diferentes períodos
            return_periods = [10, 50, 100]
            return_levels = {}

            for rp in return_periods:
                # Quantil para período de retorno
                quantile = 1 - 1 / rp

                # GEV inverse CDF
                if abs(shape) > 1e-10:
                    level = loc - scale / shape * (1 - (-np.log(quantile)) ** (-shape))
                else:
                    # Gumbel (ξ = 0)
                    level = loc - scale * np.log(-np.log(quantile))

                return_levels[f'{rp}_year'] = float(level)

        except:
            shape, loc, scale = 0, 0, 1
            return_levels = {}

        return {
            'shape': float(shape),
            'location': float(loc),
            'scale': float(scale),
            'return_levels': return_levels,
            'distribution_type': 'Frechet' if shape > 0 else ('Gumbel' if shape == 0 else 'Weibull')
        }

    @staticmethod
    def calculate_extremal_index(data: np.ndarray, threshold: float) -> Dict:
        """
        Índice extremal (mede clustering de extremos)

        Equação:
        θ = lim P(X₂ ≤ u | X₁ > u)

        θ = 1: extremos independentes
        θ < 1: extremos ocorrem em clusters
        """
        # Exceedances acima do threshold
        exceedances = data > threshold
        exceedance_indices = np.where(exceedances)[0]

        if len(exceedance_indices) < 2:
            return {
                'extremal_index': 1.0,
                'num_exceedances': len(exceedance_indices),
                'clustering': False
            }

        # Calcular gaps entre exceedances
        gaps = np.diff(exceedance_indices)

        # Intervals method para estimar θ
        # θ ≈ (número de clusters) / (número de exceedances)

        # Clusters: gaps > median(gaps)
        median_gap = np.median(gaps)
        num_clusters = np.sum(gaps > median_gap) + 1

        theta = num_clusters / len(exceedance_indices)

        return {
            'extremal_index': float(theta),
            'num_exceedances': int(len(exceedance_indices)),
            'num_clusters': int(num_clusters),
            'clustering': bool(theta < 0.8),
            'mean_cluster_size': float(len(exceedance_indices) / num_clusters)
        }


# ============================================================================
# MULTIVARIATE ANALYSIS - Análise Multivariada
# ============================================================================

class MultivariateAnalyzer:
    """
    Análise estatística multivariada
    """

    @staticmethod
    def principal_component_analysis(data: np.ndarray) -> Dict:
        """
        PCA (Principal Component Analysis)

        Equação:
        C = XᵀX / (n - 1)  (matriz de covariância)
        C = VΛVᵀ  (decomposição espectral)

        Onde V = eigenvectors, Λ = eigenvalues
        """
        if data.shape[0] < 2:
            return {
                'explained_variance': [],
                'cumulative_variance': [],
                'num_components': 0
            }

        # Centralizar
        data_centered = data - np.mean(data, axis=0)

        # Covariância
        cov_matrix = np.cov(data_centered.T)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Ordenar por eigenvalue (descendente)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Variância explicada
        total_var = np.sum(eigenvalues)
        explained_var = eigenvalues / total_var if total_var > 0 else eigenvalues

        # Variância cumulativa
        cumulative_var = np.cumsum(explained_var)

        # Número de componentes para 95% variância
        num_components = np.sum(cumulative_var < 0.95) + 1

        return {
            'explained_variance': explained_var.tolist(),
            'cumulative_variance': cumulative_var.tolist(),
            'eigenvalues': eigenvalues.tolist(),
            'num_components_95': int(num_components),
            'total_components': len(eigenvalues)
        }

    @staticmethod
    def hotelling_t2_test(
        sample1: np.ndarray,
        sample2: np.ndarray
    ) -> Dict:
        """
        Hotelling's T² test (MANOVA)

        Testa se duas amostras multivariadas têm mesma média

        Equação:
        T² = n₁n₂/(n₁+n₂) * (x̄₁ - x̄₂)ᵀ S⁻¹ (x̄₁ - x̄₂)

        Onde S = pooled covariance matrix
        """
        n1 = len(sample1)
        n2 = len(sample2)

        if n1 < 2 or n2 < 2:
            return {
                't2_statistic': 0.0,
                'f_statistic': 0.0,
                'p_value': 1.0,
                'reject_null': False
            }

        # Médias
        mean1 = np.mean(sample1, axis=0)
        mean2 = np.mean(sample2, axis=0)
        mean_diff = mean1 - mean2

        # Pooled covariance
        cov1 = np.cov(sample1.T)
        cov2 = np.cov(sample2.T)

        if sample1.ndim == 1:
            cov1 = np.array([[cov1]])
            cov2 = np.array([[cov2]])

        pooled_cov = ((n1 - 1) * cov1 + (n2 - 1) * cov2) / (n1 + n2 - 2)

        # Regularização
        pooled_cov += np.eye(pooled_cov.shape[0]) * 1e-6

        try:
            inv_cov = np.linalg.inv(pooled_cov)
        except:
            inv_cov = np.linalg.pinv(pooled_cov)

        # T² statistic
        t2 = (n1 * n2) / (n1 + n2) * mean_diff.T @ inv_cov @ mean_diff

        # Convert to F-statistic
        p = mean_diff.shape[0]  # Dimensão
        f_stat = (n1 + n2 - p - 1) / ((n1 + n2 - 2) * p) * t2

        # P-value
        df1 = p
        df2 = n1 + n2 - p - 1

        if df2 > 0:
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        else:
            p_value = 1.0

        return {
            't2_statistic': float(t2),
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'reject_null': bool(p_value < 0.05)
        }


# ============================================================================
# TESTE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTE DE ANÁLISE ESTATÍSTICA AVANÇADA")
    print("=" * 70)

    # 1. Distribution Analysis
    print("\n1. Distribution Fitting:")

    # Dados normais
    data = np.random.normal(loc=50, scale=10, size=1000)

    dist_analyzer = DistributionAnalyzer()
    best_fit = dist_analyzer.fit_best_distribution(data)

    print(f"  Best distribution: {best_fit.distribution}")
    print(f"  Parameters: {best_fit.parameters}")
    print(f"  KS p-value: {best_fit.p_value:.4f}")
    print(f"  AIC: {best_fit.aic:.2f}")

    # 2. Time Series Analysis
    print("\n2. Time Series Analysis:")

    # Série temporal com tendência
    t = np.linspace(0, 10, 100)
    ts_data = 0.5 * t + np.sin(2 * np.pi * 0.5 * t) + np.random.randn(100) * 0.5

    ts_analyzer = TimeSeriesAnalyzer()

    trend_result = ts_analyzer.detect_trend(ts_data)
    print(f"  Has trend: {trend_result['has_trend']}")
    print(f"  Direction: {trend_result['trend_direction']}")
    print(f"  Slope: {trend_result['slope']:.4f}")

    # 3. Change Point Detection
    print("\n3. Change Point Detection:")

    # Dados com change point
    data1 = np.random.normal(0, 1, 50)
    data2 = np.random.normal(5, 1, 50)
    cp_data = np.concatenate([data1, data2])

    cp_detector = ChangePointDetector()
    cp_result = cp_detector.detect_change_points(cp_data)

    print(f"  Change points: {cp_result['change_points']}")
    print(f"  Number of segments: {cp_result['num_segments']}")

    # 4. Extreme Value Analysis
    print("\n4. Extreme Value Analysis:")

    # Dados com extremos
    extreme_data = np.random.exponential(scale=2, size=1000)

    eva_analyzer = ExtremeValueAnalyzer()
    gev_result = eva_analyzer.fit_gev(extreme_data)

    print(f"  GEV shape: {gev_result['shape']:.4f}")
    print(f"  Distribution type: {gev_result['distribution_type']}")
    if gev_result['return_levels']:
        print(f"  100-year return level: {gev_result['return_levels'].get('100_year', 'N/A'):.2f}")

    # 5. Multivariate Analysis
    print("\n5. Multivariate Analysis:")

    # Dados multivariados
    mv_data = np.random.multivariate_normal(
        mean=[0, 0, 0],
        cov=[[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]],
        size=100
    )

    mv_analyzer = MultivariateAnalyzer()
    pca_result = mv_analyzer.principal_component_analysis(mv_data)

    print(f"  Explained variance: {[f'{v:.2%}' for v in pca_result['explained_variance'][:3]]}")
    print(f"  Components for 95%: {pca_result['num_components_95']}")
