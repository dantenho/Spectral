"""
Métricas Avançadas de Qualidade - Quality Assessment

Implementa métricas sofisticadas para avaliar qualidade de dados sensoriais
"""

import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from collections import deque


# ============================================================================
# DATA QUALITY METRICS
# ============================================================================

@dataclass
class QualityReport:
    """Relatório completo de qualidade"""
    overall_score: float  # [0-1], 1 = perfeito
    snr: float  # Signal-to-Noise Ratio (dB)
    thd: float  # Total Harmonic Distortion
    crest_factor: float
    peak_to_average_ratio: float
    dynamic_range: float
    effective_bits: float
    jitter: float
    drift_rate: float
    stability_score: float
    completeness: float  # Fração de dados não-nulos
    consistency: float  # Consistência temporal
    issues: List[str]


class DataQualityAnalyzer:
    """
    Analisador completo de qualidade de dados

    Implementa múltiplas métricas de qualidade para sinais sensoriais
    """

    def __init__(
        self,
        sample_rate: float = 100.0,
        window_size: int = 1000
    ):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def analyze(self, data: np.ndarray) -> QualityReport:
        """
        Análise completa de qualidade

        Args:
            data: Array de dados temporais

        Returns:
            QualityReport com todas as métricas
        """
        issues = []

        # Adicionar ao buffer
        for val in data:
            self.buffer.append(val)

        signal_data = np.array(list(self.buffer))

        # 1. SNR (Signal-to-Noise Ratio)
        snr = self._calculate_snr(signal_data)
        if snr < 10:
            issues.append(f"SNR baixo ({snr:.1f} dB)")

        # 2. THD (Total Harmonic Distortion)
        thd = self._calculate_thd(signal_data)
        if thd > 0.1:  # 10%
            issues.append(f"THD alto ({thd*100:.1f}%)")

        # 3. Crest Factor
        crest_factor = self._calculate_crest_factor(signal_data)

        # 4. Peak-to-Average Ratio
        par = self._calculate_par(signal_data)

        # 5. Dynamic Range
        dynamic_range = self._calculate_dynamic_range(signal_data)

        # 6. Effective Bits (ENOB)
        enob = self._calculate_enob(snr)

        # 7. Jitter (variação temporal)
        jitter = self._calculate_jitter(signal_data)
        if jitter > 0.01:
            issues.append(f"Jitter alto ({jitter:.4f})")

        # 8. Drift Rate
        drift_rate = self._calculate_drift_rate(signal_data)
        if abs(drift_rate) > 0.001:
            issues.append(f"Drift detectado ({drift_rate:.5f})")

        # 9. Stability Score
        stability = self._calculate_stability(signal_data)
        if stability < 0.7:
            issues.append(f"Baixa estabilidade ({stability:.2f})")

        # 10. Completeness
        completeness = np.sum(~np.isnan(signal_data)) / len(signal_data)
        if completeness < 0.95:
            issues.append(f"Dados incompletos ({completeness*100:.1f}%)")

        # 11. Consistency
        consistency = self._calculate_consistency(signal_data)
        if consistency < 0.8:
            issues.append(f"Inconsistência temporal ({consistency:.2f})")

        # Overall Score (média ponderada)
        overall_score = self._calculate_overall_score(
            snr, thd, stability, completeness, consistency
        )

        return QualityReport(
            overall_score=overall_score,
            snr=snr,
            thd=thd,
            crest_factor=crest_factor,
            peak_to_average_ratio=par,
            dynamic_range=dynamic_range,
            effective_bits=enob,
            jitter=jitter,
            drift_rate=drift_rate,
            stability_score=stability,
            completeness=completeness,
            consistency=consistency,
            issues=issues
        )

    def _calculate_snr(self, signal_data: np.ndarray) -> float:
        """
        SNR (Signal-to-Noise Ratio)

        Equação:
        SNR = 10 * log₁₀(P_signal / P_noise)
        """
        # Estimar sinal (componente de baixa frequência)
        # Usar filtro passa-baixa simples
        b, a = signal.butter(4, 0.1, btype='low')
        signal_est = signal.filtfilt(b, a, signal_data)

        # Ruído = original - sinal estimado
        noise = signal_data - signal_est

        # Potências
        signal_power = np.mean(signal_est ** 2)
        noise_power = np.mean(noise ** 2)

        if noise_power < 1e-10:
            return 100.0  # SNR muito alto

        snr_db = 10 * np.log10(signal_power / noise_power)

        return float(snr_db)

    def _calculate_thd(self, signal_data: np.ndarray) -> float:
        """
        THD (Total Harmonic Distortion)

        Equação:
        THD = √(P₂² + P₃² + ... + Pₙ²) / P₁

        Onde Pᵢ = potência da i-ésima harmônica
        """
        # FFT
        N = len(signal_data)
        fft_vals = np.abs(fft(signal_data))[:N//2]
        freqs = fftfreq(N, 1/self.sample_rate)[:N//2]

        # Encontrar fundamental (frequência dominante)
        fundamental_idx = np.argmax(fft_vals)
        fundamental_power = fft_vals[fundamental_idx] ** 2

        if fundamental_power < 1e-10:
            return 0.0

        # Harmonics (2f, 3f, 4f, 5f)
        harmonic_power = 0.0
        for n in range(2, 6):
            harmonic_freq = freqs[fundamental_idx] * n

            # Encontrar bin mais próximo
            harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))

            if harmonic_idx < len(fft_vals):
                harmonic_power += fft_vals[harmonic_idx] ** 2

        thd = np.sqrt(harmonic_power) / np.sqrt(fundamental_power)

        return float(thd)

    def _calculate_crest_factor(self, signal_data: np.ndarray) -> float:
        """
        Crest Factor = |peak| / RMS

        Mede quão "spiky" é o sinal
        """
        peak = np.max(np.abs(signal_data))
        rms = np.sqrt(np.mean(signal_data ** 2))

        if rms < 1e-10:
            return 1.0

        crest_factor = peak / rms

        return float(crest_factor)

    def _calculate_par(self, signal_data: np.ndarray) -> float:
        """
        Peak-to-Average Ratio (PAR)

        PAR = peak² / mean(x²)
        """
        peak_power = np.max(signal_data) ** 2
        avg_power = np.mean(signal_data ** 2)

        if avg_power < 1e-10:
            return 1.0

        par = peak_power / avg_power

        return float(par)

    def _calculate_dynamic_range(self, signal_data: np.ndarray) -> float:
        """
        Dynamic Range (dB)

        DR = 20 * log₁₀(max / min)
        """
        max_val = np.max(np.abs(signal_data))

        # Usar percentil 1 como "min" para evitar outliers
        min_val = np.percentile(np.abs(signal_data), 1)

        if min_val < 1e-10:
            min_val = 1e-10

        dr_db = 20 * np.log10(max_val / min_val)

        return float(dr_db)

    def _calculate_enob(self, snr_db: float) -> float:
        """
        ENOB (Effective Number of Bits)

        Equação:
        ENOB = (SNR - 1.76) / 6.02

        Derivado de: SNR_ideal = 6.02*N + 1.76 (para ADC de N bits)
        """
        enob = (snr_db - 1.76) / 6.02

        return float(max(0, enob))

    def _calculate_jitter(self, signal_data: np.ndarray) -> float:
        """
        Jitter - Variação temporal nos cruzamentos por zero

        Mede instabilidade temporal
        """
        # Encontrar zero-crossings
        zero_crossings = np.where(np.diff(np.sign(signal_data)))[0]

        if len(zero_crossings) < 2:
            return 0.0

        # Calcular períodos entre crossings
        periods = np.diff(zero_crossings)

        # Jitter = std dos períodos / mean dos períodos
        mean_period = np.mean(periods)

        if mean_period < 1e-10:
            return 0.0

        jitter = np.std(periods) / mean_period

        return float(jitter)

    def _calculate_drift_rate(self, signal_data: np.ndarray) -> float:
        """
        Drift Rate - Taxa de mudança linear da média

        Equação:
        drift = slope / mean

        Usa regressão linear
        """
        if len(signal_data) < 10:
            return 0.0

        x = np.arange(len(signal_data))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, signal_data)

        mean_val = np.mean(signal_data)

        if abs(mean_val) < 1e-10:
            return 0.0

        drift_rate = slope / abs(mean_val)

        return float(drift_rate)

    def _calculate_stability(self, signal_data: np.ndarray) -> float:
        """
        Stability Score baseado em Allan Variance simplificada

        Score alto = sinal estável
        """
        # Calcular variância móvel
        window = 10

        if len(signal_data) < window * 2:
            return 1.0

        variances = []
        for i in range(0, len(signal_data) - window, window):
            window_data = signal_data[i:i+window]
            variances.append(np.var(window_data))

        if not variances:
            return 1.0

        # Estabilidade = inverso da variação das variâncias
        variance_of_variances = np.var(variances)
        mean_variance = np.mean(variances)

        if mean_variance < 1e-10:
            return 1.0

        stability = 1.0 / (1.0 + variance_of_variances / mean_variance)

        return float(stability)

    def _calculate_consistency(self, signal_data: np.ndarray) -> float:
        """
        Consistency - Autocorrelação com lag=1

        Alta autocorrelação = consistência temporal
        """
        if len(signal_data) < 2:
            return 1.0

        # Normalizar
        normalized = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-10)

        # Autocorrelação lag=1
        autocorr = np.corrcoef(normalized[:-1], normalized[1:])[0, 1]

        # Converter para [0, 1]
        consistency = (autocorr + 1) / 2

        return float(consistency)

    def _calculate_overall_score(
        self,
        snr: float,
        thd: float,
        stability: float,
        completeness: float,
        consistency: float
    ) -> float:
        """
        Overall Quality Score (média ponderada)
        """
        # Normalizar SNR para [0, 1]
        snr_norm = np.clip(snr / 40.0, 0, 1)  # 40 dB = perfeito

        # Normalizar THD para [0, 1]
        thd_norm = 1.0 - np.clip(thd / 0.1, 0, 1)  # 0.1 = 10% = ruim

        # Pesos
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]
        scores = [snr_norm, thd_norm, stability, completeness, consistency]

        overall = np.average(scores, weights=weights)

        return float(overall)


# ============================================================================
# STATISTICAL QUALITY METRICS
# ============================================================================

class StatisticalQualityMetrics:
    """
    Métricas estatísticas avançadas para qualidade
    """

    @staticmethod
    def calculate_zscore_quality(data: np.ndarray) -> Dict:
        """
        Qualidade baseada em Z-scores

        Detecta outliers usando distribuição normal
        """
        mean = np.mean(data)
        std = np.std(data)

        if std < 1e-10:
            z_scores = np.zeros_like(data)
        else:
            z_scores = (data - mean) / std

        # Outliers (|z| > 3)
        outliers = np.abs(z_scores) > 3
        outlier_ratio = np.sum(outliers) / len(data)

        # Quality score
        quality = 1.0 - np.clip(outlier_ratio * 10, 0, 1)

        return {
            'quality': float(quality),
            'outlier_ratio': float(outlier_ratio),
            'mean_zscore': float(np.mean(np.abs(z_scores))),
            'max_zscore': float(np.max(np.abs(z_scores)))
        }

    @staticmethod
    def calculate_normality_quality(data: np.ndarray) -> Dict:
        """
        Qualidade baseada em normalidade (Shapiro-Wilk test)

        Dados "bons" tendem a seguir distribuição normal
        """
        if len(data) < 3:
            return {'quality': 1.0, 'p_value': 1.0, 'is_normal': True}

        # Shapiro-Wilk test
        try:
            statistic, p_value = stats.shapiro(data)
        except:
            return {'quality': 0.5, 'p_value': 0.5, 'is_normal': False}

        # p > 0.05 = normal
        is_normal = p_value > 0.05

        # Quality = p-value (quanto maior, mais normal)
        quality = float(np.clip(p_value, 0, 1))

        return {
            'quality': quality,
            'p_value': float(p_value),
            'is_normal': bool(is_normal),
            'statistic': float(statistic)
        }

    @staticmethod
    def calculate_stationarity_quality(data: np.ndarray) -> Dict:
        """
        Qualidade baseada em estacionariedade

        Divide em janelas e compara estatísticas
        """
        if len(data) < 100:
            return {'quality': 1.0, 'is_stationary': True}

        # Dividir em 4 janelas
        chunk_size = len(data) // 4
        chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(4)]

        # Calcular médias e variâncias
        means = [np.mean(chunk) for chunk in chunks]
        vars = [np.var(chunk) for chunk in chunks]

        # Variação das médias e variâncias
        mean_var = np.var(means)
        var_var = np.var(vars)

        # Normalizar
        mean_stability = 1.0 / (1.0 + mean_var)
        var_stability = 1.0 / (1.0 + var_var)

        # Quality = média
        quality = (mean_stability + var_stability) / 2

        # É estacionário se quality > 0.8
        is_stationary = quality > 0.8

        return {
            'quality': float(quality),
            'is_stationary': bool(is_stationary),
            'mean_variance': float(mean_var),
            'variance_variance': float(var_var)
        }


# ============================================================================
# MULTI-SENSOR QUALITY
# ============================================================================

class MultiSensorQualityAnalyzer:
    """
    Analisa qualidade considerando múltiplos sensores
    """

    @staticmethod
    def analyze_correlation_quality(
        sensor_data: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Qualidade baseada em correlações entre sensores

        Sensores relacionados devem ter correlação esperada
        """
        if len(sensor_data) < 2:
            return {'quality': 1.0, 'correlations': {}}

        sensor_names = list(sensor_data.keys())
        correlations = {}

        for i in range(len(sensor_names)):
            for j in range(i + 1, len(sensor_names)):
                name1 = sensor_names[i]
                name2 = sensor_names[j]

                data1 = sensor_data[name1]
                data2 = sensor_data[name2]

                # Garantir mesmo tamanho
                min_len = min(len(data1), len(data2))
                data1 = data1[:min_len]
                data2 = data2[:min_len]

                # Correlação
                if len(data1) > 1:
                    corr = np.corrcoef(data1, data2)[0, 1]
                    correlations[f"{name1}-{name2}"] = float(corr)

        # Quality = média das correlações absolutas
        if correlations:
            avg_corr = np.mean([abs(c) for c in correlations.values()])
            quality = float(avg_corr)
        else:
            quality = 1.0

        return {
            'quality': quality,
            'correlations': correlations
        }

    @staticmethod
    def detect_sensor_failures(
        sensor_data: Dict[str, np.ndarray]
    ) -> Dict:
        """
        Detecta falhas de sensores

        Critérios:
        - Valores constantes (stuck)
        - Valores saturados
        - Descontinuidades
        """
        failures = {}

        for name, data in sensor_data.items():
            issues = []

            if len(data) < 10:
                continue

            # 1. Stuck (valores constantes)
            unique_ratio = len(np.unique(data)) / len(data)
            if unique_ratio < 0.01:
                issues.append("stuck")

            # 2. Saturado (muitos valores no limite)
            sorted_data = np.sort(data)
            min_count = np.sum(data == sorted_data[0])
            max_count = np.sum(data == sorted_data[-1])

            saturation_ratio = (min_count + max_count) / len(data)
            if saturation_ratio > 0.1:  # 10% saturado
                issues.append("saturated")

            # 3. Descontinuidades (grandes saltos)
            diffs = np.abs(np.diff(data))
            median_diff = np.median(diffs)

            if median_diff > 0:
                large_jumps = np.sum(diffs > median_diff * 10)
                jump_ratio = large_jumps / len(diffs)

                if jump_ratio > 0.05:  # 5% jumps
                    issues.append("discontinuous")

            if issues:
                failures[name] = issues

        overall_health = 1.0 - (len(failures) / len(sensor_data))

        return {
            'overall_health': float(overall_health),
            'failures': failures,
            'healthy_sensors': len(sensor_data) - len(failures),
            'total_sensors': len(sensor_data)
        }


# ============================================================================
# TESTE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTE DE MÉTRICAS DE QUALIDADE")
    print("=" * 70)

    # 1. Data Quality Analyzer
    print("\n1. Data Quality Analyzer:")

    # Sinal de teste com ruído
    t = np.linspace(0, 10, 1000)
    clean_signal = np.sin(2 * np.pi * 1 * t)
    noisy_signal = clean_signal + 0.1 * np.random.randn(1000)

    analyzer = DataQualityAnalyzer(sample_rate=100)
    report = analyzer.analyze(noisy_signal)

    print(f"  Overall Score: {report.overall_score:.2%}")
    print(f"  SNR: {report.snr:.2f} dB")
    print(f"  THD: {report.thd*100:.2f}%")
    print(f"  Stability: {report.stability_score:.2%}")
    print(f"  ENOB: {report.effective_bits:.2f} bits")

    if report.issues:
        print(f"  Issues: {', '.join(report.issues)}")

    # 2. Statistical Quality
    print("\n2. Statistical Quality:")

    stat_metrics = StatisticalQualityMetrics()

    z_quality = stat_metrics.calculate_zscore_quality(noisy_signal)
    print(f"  Z-score Quality: {z_quality['quality']:.2%}")
    print(f"  Outlier Ratio: {z_quality['outlier_ratio']*100:.2f}%")

    # 3. Multi-Sensor Quality
    print("\n3. Multi-Sensor Quality:")

    multi_analyzer = MultiSensorQualityAnalyzer()

    # Dados de múltiplos sensores
    sensors = {
        'accel_x': np.sin(2 * np.pi * 1 * t) + 0.1 * np.random.randn(1000),
        'accel_y': np.cos(2 * np.pi * 1 * t) + 0.1 * np.random.randn(1000),
        'gyro_x': np.sin(2 * np.pi * 2 * t) + 0.05 * np.random.randn(1000)
    }

    corr_quality = multi_analyzer.analyze_correlation_quality(sensors)
    print(f"  Correlation Quality: {corr_quality['quality']:.2%}")

    health = multi_analyzer.detect_sensor_failures(sensors)
    print(f"  Overall Health: {health['overall_health']:.2%}")
    print(f"  Healthy Sensors: {health['healthy_sensors']}/{health['total_sensors']}")
