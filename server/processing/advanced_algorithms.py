"""
Algoritmos Avançados de Precisão - Servidor Python

Implementa equações matemáticas e estatísticas sofisticadas
"""

import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from collections import deque


# ============================================================================
# CUSUM (Cumulative Sum Control Chart) - Detecção de Mudanças
# ============================================================================

class CUSUMDetector:
    """
    CUSUM - Cumulative Sum para detectar mudanças sutis na média

    Equações:
    S⁺ᵢ = max(0, S⁺ᵢ₋₁ + (xᵢ - μ₀ - k))
    S⁻ᵢ = max(0, S⁻ᵢ₋₁ - (xᵢ - μ₀ - k))

    Detecta mudança se S⁺ > h ou S⁻ > h

    Onde:
    - k = slack parameter (tipicamente k = σ/2)
    - h = threshold (tipicamente h = 5σ)
    - μ₀ = valor alvo
    """

    def __init__(
        self,
        target: float = 0.0,
        k: Optional[float] = None,
        h: Optional[float] = None,
        reset_after_detection: bool = True
    ):
        self.target = target
        self.k = k
        self.h = h
        self.reset_after_detection = reset_after_detection

        self.s_plus = 0.0
        self.s_minus = 0.0
        self.samples = []

        # Estimativa inicial de μ e σ
        self.mean = target
        self.std = 1.0

    def update(self, value: float) -> Dict:
        """Atualiza CUSUM com nova medição"""

        self.samples.append(value)

        # Estimar μ e σ das primeiras amostras
        if len(self.samples) >= 10:
            self.mean = np.mean(self.samples[-100:])
            self.std = np.std(self.samples[-100:]) + 1e-6

            # Usar padrões para k e h se não especificado
            if self.k is None:
                self.k = self.std / 2

            if self.h is None:
                self.h = 5 * self.std

        # CUSUM
        deviation = value - self.mean
        self.s_plus = max(0, self.s_plus + (deviation - self.k))
        self.s_minus = max(0, self.s_minus - (deviation + self.k))

        # Detecção
        upward_shift = self.s_plus > self.h
        downward_shift = self.s_minus > self.h

        change_detected = upward_shift or downward_shift

        if change_detected and self.reset_after_detection:
            self.s_plus = 0.0
            self.s_minus = 0.0

        return {
            's_plus': self.s_plus,
            's_minus': self.s_minus,
            'change_detected': change_detected,
            'upward_shift': upward_shift,
            'downward_shift': downward_shift,
            'z_score': deviation / self.std if self.std > 0 else 0
        }


# ============================================================================
# EWMA (Exponentially Weighted Moving Average) - Detecção Suave
# ============================================================================

class EWMADetector:
    """
    EWMA - Exponentially Weighted Moving Average

    Equação:
    Zᵢ = λ * xᵢ + (1 - λ) * Zᵢ₋₁

    Controle:
    UCL = μ₀ + L * σ * sqrt(λ/(2-λ) * (1-(1-λ)^(2i)))
    LCL = μ₀ - L * σ * sqrt(λ/(2-λ) * (1-(1-λ)^(2i)))

    Onde:
    - λ = peso (tipicamente 0.2)
    - L = largura dos limites de controle (tipicamente 3)
    """

    def __init__(
        self,
        lambda_: float = 0.2,
        L: float = 3.0,
        target: float = 0.0
    ):
        self.lambda_ = lambda_
        self.L = L
        self.target = target

        self.z = target
        self.i = 0
        self.samples = []

        self.mean = target
        self.std = 1.0

    def update(self, value: float) -> Dict:
        """Atualiza EWMA"""

        self.samples.append(value)
        self.i += 1

        # Estimar μ e σ
        if len(self.samples) >= 10:
            self.mean = np.mean(self.samples[-100:])
            self.std = np.std(self.samples[-100:]) + 1e-6

        # EWMA
        self.z = self.lambda_ * value + (1 - self.lambda_) * self.z

        # Limites de controle
        factor = np.sqrt(
            (self.lambda_ / (2 - self.lambda_)) *
            (1 - (1 - self.lambda_) ** (2 * self.i))
        )

        ucl = self.mean + self.L * self.std * factor
        lcl = self.mean - self.L * self.std * factor

        # Detecção
        out_of_control = (self.z > ucl) or (self.z < lcl)

        return {
            'ewma': self.z,
            'ucl': ucl,
            'lcl': lcl,
            'out_of_control': out_of_control,
            'deviation': abs(self.z - self.mean) / self.std
        }


# ============================================================================
# CROSS-CORRELATION - Correlação Cruzada Multi-Sensores
# ============================================================================

class CrossCorrelationAnalyzer:
    """
    Análise de correlação cruzada entre múltiplos sensores

    Equação de correlação cruzada normalizada:
    ρ(τ) = Σ[(xᵢ - μₓ)(yᵢ₊τ - μᵧ)] / (σₓ * σᵧ * N)

    Onde:
    - τ = lag (atraso temporal)
    - ρ ∈ [-1, 1]
    """

    def __init__(self, max_lag: int = 50):
        self.max_lag = max_lag

    def correlate(
        self,
        signal1: np.ndarray,
        signal2: np.ndarray
    ) -> Dict:
        """Calcula correlação cruzada"""

        # Normalizar sinais
        s1_norm = (signal1 - np.mean(signal1)) / (np.std(signal1) + 1e-10)
        s2_norm = (signal2 - np.mean(signal2)) / (np.std(signal2) + 1e-10)

        # Correlação cruzada
        correlation = signal.correlate(s1_norm, s2_norm, mode='same')
        correlation = correlation / len(signal1)

        # Encontrar lag de máxima correlação
        center = len(correlation) // 2
        lags = np.arange(-self.max_lag, self.max_lag + 1)
        valid_corr = correlation[center - self.max_lag:center + self.max_lag + 1]

        max_corr_idx = np.argmax(np.abs(valid_corr))
        max_corr = valid_corr[max_corr_idx]
        best_lag = lags[max_corr_idx]

        # Pearson correlation (lag = 0)
        pearson_corr = np.corrcoef(signal1, signal2)[0, 1]

        # Significância estatística (teste de hipótese)
        n = len(signal1)
        if abs(pearson_corr) > 0:
            t_stat = pearson_corr * np.sqrt((n - 2) / (1 - pearson_corr ** 2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        else:
            t_stat = 0
            p_value = 1.0

        return {
            'max_correlation': float(max_corr),
            'best_lag': int(best_lag),
            'pearson_correlation': float(pearson_corr),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        }


# ============================================================================
# POWER SPECTRAL DENSITY - Análise de Densidade Espectral de Potência
# ============================================================================

class PowerSpectralDensityAnalyzer:
    """
    Análise PSD (Power Spectral Density) usando método de Welch

    Equação de PSD (método de Welch):
    Pₓₓ(f) = (1/K) * Σ |Xₖ(f)|²

    Onde:
    - K = número de segmentos
    - Xₖ(f) = FFT do segmento k
    """

    def __init__(
        self,
        sample_rate: float = 100.0,
        nperseg: int = 256,
        noverlap: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.nperseg = nperseg
        self.noverlap = noverlap or nperseg // 2

    def analyze(self, signal_data: np.ndarray) -> Dict:
        """Calcula PSD e métricas derivadas"""

        # Welch's method
        freqs, psd = signal.welch(
            signal_data,
            fs=self.sample_rate,
            nperseg=self.nperseg,
            noverlap=self.noverlap
        )

        # Potência total
        total_power = np.trapz(psd, freqs)

        # Frequência dominante
        dominant_freq_idx = np.argmax(psd)
        dominant_freq = freqs[dominant_freq_idx]
        dominant_power = psd[dominant_freq_idx]

        # Largura de banda (onde está 90% da potência)
        cumulative_power = np.cumsum(psd) / total_power
        bw_lower_idx = np.where(cumulative_power >= 0.05)[0][0]
        bw_upper_idx = np.where(cumulative_power >= 0.95)[0][0]
        bandwidth = freqs[bw_upper_idx] - freqs[bw_lower_idx]

        # Centróide espectral (centro de massa do espectro)
        spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)

        # Entropia espectral (dispersão da energia)
        psd_norm = psd / (np.sum(psd) + 1e-10)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

        return {
            'frequencies': freqs.tolist(),
            'psd': psd.tolist(),
            'total_power': float(total_power),
            'dominant_frequency': float(dominant_freq),
            'dominant_power': float(dominant_power),
            'bandwidth': float(bandwidth),
            'spectral_centroid': float(spectral_centroid),
            'spectral_entropy': float(spectral_entropy)
        }


# ============================================================================
# BAYESIAN INFERENCE - Inferência Bayesiana para Classificação
# ============================================================================

class BayesianClassifier:
    """
    Classificação Bayesiana com atualização de crenças

    Teorema de Bayes:
    P(H|E) = P(E|H) * P(H) / P(E)

    Onde:
    - P(H|E) = probabilidade posterior (após evidência)
    - P(E|H) = verossimilhança (likelihood)
    - P(H) = probabilidade prior
    - P(E) = evidência (normalização)
    """

    def __init__(self, classes: List[str]):
        self.classes = classes
        self.num_classes = len(classes)

        # Priors uniformes
        self.priors = {c: 1.0 / self.num_classes for c in classes}

        # Likelihoods observados
        self.likelihoods = {c: [] for c in classes}

    def update(self, evidence: Dict[str, float]) -> Dict:
        """
        Atualiza probabilidades posteriores com nova evidência

        Args:
            evidence: Dict[class_name, likelihood]

        Returns:
            Dict com posteriores e classe predita
        """

        # Calcular posteriors usando Bayes
        posteriors = {}
        evidence_sum = 0.0

        for class_name in self.classes:
            likelihood = evidence.get(class_name, 1e-6)
            prior = self.priors[class_name]

            # P(H|E) ∝ P(E|H) * P(H)
            posterior = likelihood * prior
            posteriors[class_name] = posterior
            evidence_sum += posterior

        # Normalizar (P(E))
        if evidence_sum > 0:
            posteriors = {k: v / evidence_sum for k, v in posteriors.items()}

        # Atualizar priors para próxima iteração
        self.priors = posteriors.copy()

        # Classe predita
        predicted_class = max(posteriors, key=posteriors.get)
        confidence = posteriors[predicted_class]

        return {
            'posteriors': posteriors,
            'predicted_class': predicted_class,
            'confidence': confidence
        }


# ============================================================================
# MAHALANOBIS DISTANCE - Distância de Mahalanobis
# ============================================================================

class MahalanobisDetector:
    """
    Distância de Mahalanobis para detecção de anomalias multivariadas

    Equação:
    D²(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)

    Onde:
    - μ = vetor de médias
    - Σ = matriz de covariância
    - D² segue distribuição χ² com p graus de liberdade
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.samples = []
        self.mean = None
        self.cov = None
        self.inv_cov = None

    def fit(self, data: np.ndarray):
        """Treina detector com dados normais"""

        self.samples = data
        self.mean = np.mean(data, axis=0)
        self.cov = np.cov(data.T)

        # Adicionar regularização para evitar singularidade
        self.cov += np.eye(self.cov.shape[0]) * 1e-6

        try:
            self.inv_cov = np.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            # Se singular, usar pseudo-inversa
            self.inv_cov = np.linalg.pinv(self.cov)

    def detect(self, x: np.ndarray) -> Dict:
        """Detecta se x é anomalia"""

        if self.mean is None or self.inv_cov is None:
            return {'is_anomaly': False, 'distance': 0.0, 'p_value': 1.0}

        # Calcular distância de Mahalanobis
        diff = x - self.mean
        d_squared = diff.T @ self.inv_cov @ diff

        distance = np.sqrt(d_squared)

        # P-value usando distribuição χ²
        p = len(x)
        p_value = 1 - stats.chi2.cdf(d_squared, df=p)

        # Anomalia se distância > threshold * escala típica
        is_anomaly = distance > self.threshold

        return {
            'is_anomaly': bool(is_anomaly),
            'distance': float(distance),
            'p_value': float(p_value),
            'threshold': float(self.threshold)
        }


# ============================================================================
# ENSEMBLE CONFIDENCE - Confiança de Ensemble Avançada
# ============================================================================

class EnsembleConfidenceCalculator:
    """
    Calcula métricas avançadas de confiança para ensembles

    Implementa múltiplas métricas:
    1. Shannon Entropy
    2. Gini Impurity
    3. Margin (diferença entre top 2 classes)
    4. Variation Ratio
    """

    @staticmethod
    def shannon_entropy(probabilities: np.ndarray) -> float:
        """
        Shannon Entropy: H = -Σ p(x) * log(p(x))

        Menor entropia = mais confiante
        """
        # Evitar log(0)
        probs = np.clip(probabilities, 1e-10, 1.0)
        entropy = -np.sum(probs * np.log2(probs))

        # Normalizar para [0, 1]
        max_entropy = np.log2(len(probabilities))
        normalized = entropy / max_entropy if max_entropy > 0 else 0

        # Confiança = 1 - entropy_normalizada
        confidence = 1.0 - normalized

        return float(confidence)

    @staticmethod
    def gini_impurity(probabilities: np.ndarray) -> float:
        """
        Gini Impurity: G = 1 - Σ p(x)²

        Menor impureza = mais confiante
        """
        gini = 1.0 - np.sum(probabilities ** 2)

        # Normalizar para [0, 1]
        num_classes = len(probabilities)
        max_gini = 1.0 - 1.0 / num_classes

        if max_gini > 0:
            normalized = gini / max_gini
        else:
            normalized = 0

        # Confiança = 1 - gini_normalizado
        confidence = 1.0 - normalized

        return float(confidence)

    @staticmethod
    def margin(probabilities: np.ndarray) -> float:
        """
        Margin: diferença entre top 2 probabilidades

        Maior margin = mais confiante
        """
        sorted_probs = np.sort(probabilities)[::-1]

        if len(sorted_probs) >= 2:
            margin_value = sorted_probs[0] - sorted_probs[1]
        else:
            margin_value = sorted_probs[0]

        return float(margin_value)

    @staticmethod
    def variation_ratio(predictions: List[int]) -> float:
        """
        Variation Ratio: proporção de não-modais

        VR = 1 - (count(mode) / N)

        Menor VR = mais acordo = mais confiante
        """
        if not predictions:
            return 0.0

        # Contar frequências
        from collections import Counter
        counts = Counter(predictions)

        # Moda
        mode_count = counts.most_common(1)[0][1]

        # Variation ratio
        vr = 1.0 - (mode_count / len(predictions))

        # Confiança = 1 - VR
        confidence = 1.0 - vr

        return float(confidence)

    def calculate_all(
        self,
        probabilities: np.ndarray,
        individual_predictions: Optional[List[int]] = None
    ) -> Dict:
        """Calcula todas as métricas de confiança"""

        metrics = {
            'entropy_confidence': self.shannon_entropy(probabilities),
            'gini_confidence': self.gini_impurity(probabilities),
            'margin': self.margin(probabilities)
        }

        if individual_predictions is not None:
            metrics['variation_confidence'] = self.variation_ratio(individual_predictions)

        # Confiança agregada (média)
        metrics['overall_confidence'] = np.mean(list(metrics.values()))

        return metrics


# ============================================================================
# TESTE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTE DE ALGORITMOS AVANÇADOS")
    print("=" * 70)

    # 1. CUSUM
    print("\n1. CUSUM Detector:")
    cusum = CUSUMDetector(target=50.0, k=0.5, h=5.0)

    data = [50, 51, 49, 50, 52, 55, 58, 60, 62, 65]  # Mudança para cima
    for val in data:
        result = cusum.update(val)
        if result['change_detected']:
            print(f"  Mudança detectada em {val}! S+ = {result['s_plus']:.2f}")

    # 2. Cross-Correlation
    print("\n2. Cross-Correlation:")
    correlator = CrossCorrelationAnalyzer()

    signal1 = np.sin(2 * np.pi * 1 * np.linspace(0, 1, 100))
    signal2 = np.sin(2 * np.pi * 1 * np.linspace(0.1, 1.1, 100))  # Com lag

    result = correlator.correlate(signal1, signal2)
    print(f"  Max correlation: {result['max_correlation']:.3f}")
    print(f"  Best lag: {result['best_lag']}")
    print(f"  Significant: {result['significant']}")

    # 3. PSD
    print("\n3. Power Spectral Density:")
    psd_analyzer = PowerSpectralDensityAnalyzer(sample_rate=100)

    signal_data = np.sin(2 * np.pi * 10 * np.linspace(0, 1, 1000))
    signal_data += 0.5 * np.random.randn(1000)

    result = psd_analyzer.analyze(signal_data)
    print(f"  Dominant frequency: {result['dominant_frequency']:.2f} Hz")
    print(f"  Bandwidth: {result['bandwidth']:.2f} Hz")
    print(f"  Spectral entropy: {result['spectral_entropy']:.3f}")

    # 4. Bayesian Classifier
    print("\n4. Bayesian Classifier:")
    bayesian = BayesianClassifier(['Normal', 'Anomalia', 'Interferência'])

    evidence = {'Normal': 0.2, 'Anomalia': 0.7, 'Interferência': 0.1}
    result = bayesian.update(evidence)

    print(f"  Predicted: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.2%}")

    # 5. Ensemble Confidence
    print("\n5. Ensemble Confidence:")
    confidence_calc = EnsembleConfidenceCalculator()

    probabilities = np.array([0.7, 0.2, 0.05, 0.05])
    predictions = [0, 0, 0, 1, 0]  # Maioria classe 0

    metrics = confidence_calc.calculate_all(probabilities, predictions)
    print(f"  Overall confidence: {metrics['overall_confidence']:.2%}")
    print(f"  Entropy confidence: {metrics['entropy_confidence']:.2%}")
    print(f"  Margin: {metrics['margin']:.3f}")
