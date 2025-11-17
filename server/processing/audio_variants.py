"""
Múltiplas variantes de processamento de áudio para detecção de anomalias

Cada variante pode ser usada separadamente ou combinada para melhorar métricas
"""

import numpy as np
import librosa
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class AudioAnalysisResult:
    """Resultado da análise de áudio"""
    variant_name: str
    anomaly_detected: bool
    confidence: float
    features: Dict[str, float]
    metadata: Dict


class AudioVariant:
    """Classe base para variantes de análise de áudio"""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.name = "base"

    def analyze(self, audio: np.ndarray) -> AudioAnalysisResult:
        """Analisa áudio e retorna resultado"""
        raise NotImplementedError


# ============================================================================
# VARIANTE 1: FFT Clássica com Análise de Frequências
# ============================================================================

class FFTVariant(AudioVariant):
    """
    Análise FFT clássica com detecção de picos em bandas específicas

    Detecta:
    - Infrassom (< 20 Hz)
    - Ultrassom (> 18 kHz)
    - Picos anômalos
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        infrasound_threshold_db: float = -40,
        ultrasound_threshold_db: float = -40,
        anomaly_threshold_db: float = -20
    ):
        super().__init__(sample_rate)
        self.name = "fft_classic"
        self.infrasound_threshold = infrasound_threshold_db
        self.ultrasound_threshold = ultrasound_threshold_db
        self.anomaly_threshold = anomaly_threshold_db

    def analyze(self, audio: np.ndarray) -> AudioAnalysisResult:
        """Análise FFT completa"""

        # FFT
        N = len(audio)
        yf = fft(audio)
        xf = fftfreq(N, 1 / self.sample_rate)

        # Apenas frequências positivas
        positive_mask = xf > 0
        xf = xf[positive_mask]
        yf = yf[positive_mask]

        # Magnitude em dB
        magnitude_db = 20 * np.log10(np.abs(yf) + 1e-10)

        # Análise de bandas
        infrasound_mask = xf < 20
        ultrasound_mask = xf > 18000
        voice_mask = (xf >= 80) & (xf <= 4000)

        features = {}

        # Infrassom
        if np.any(infrasound_mask):
            infrasound_power = np.max(magnitude_db[infrasound_mask])
            features['infrasound_power_db'] = float(infrasound_power)
            infrasound_anomaly = infrasound_power > self.infrasound_threshold
        else:
            features['infrasound_power_db'] = -np.inf
            infrasound_anomaly = False

        # Ultrassom
        if np.any(ultrasound_mask):
            ultrasound_power = np.max(magnitude_db[ultrasound_mask])
            features['ultrasound_power_db'] = float(ultrasound_power)
            ultrasound_anomaly = ultrasound_power > self.ultrasound_threshold
        else:
            features['ultrasound_power_db'] = -np.inf
            ultrasound_anomaly = False

        # Frequência dominante
        peak_idx = np.argmax(magnitude_db)
        peak_freq = float(xf[peak_idx])
        peak_power = float(magnitude_db[peak_idx])

        features['peak_frequency_hz'] = peak_freq
        features['peak_power_db'] = peak_power

        # Energia na banda de voz
        if np.any(voice_mask):
            voice_energy = np.mean(magnitude_db[voice_mask])
            features['voice_band_energy_db'] = float(voice_energy)
        else:
            features['voice_band_energy_db'] = -np.inf

        # Decisão de anomalia
        anomaly_detected = infrasound_anomaly or ultrasound_anomaly or (peak_power > self.anomaly_threshold)

        # Confidence
        confidence = 0.0
        if infrasound_anomaly:
            confidence += 0.4
        if ultrasound_anomaly:
            confidence += 0.4
        if peak_power > self.anomaly_threshold:
            confidence += 0.2

        confidence = min(confidence, 1.0)

        return AudioAnalysisResult(
            variant_name=self.name,
            anomaly_detected=anomaly_detected,
            confidence=confidence,
            features=features,
            metadata={'num_samples': N}
        )


# ============================================================================
# VARIANTE 2: STFT (Short-Time Fourier Transform) - Análise Temporal
# ============================================================================

class STFTVariant(AudioVariant):
    """
    Short-Time Fourier Transform para análise tempo-frequência

    Detecta mudanças temporais no espectro
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        window_size: int = 2048,
        hop_length: int = 512,
        transient_threshold: float = 10.0
    ):
        super().__init__(sample_rate)
        self.name = "stft_temporal"
        self.window_size = window_size
        self.hop_length = hop_length
        self.transient_threshold = transient_threshold

    def analyze(self, audio: np.ndarray) -> AudioAnalysisResult:
        """Análise STFT"""

        # STFT
        f, t, Zxx = signal.stft(
            audio,
            fs=self.sample_rate,
            window='hann',
            nperseg=self.window_size,
            noverlap=self.window_size - self.hop_length
        )

        # Magnitude do espectrograma
        magnitude = np.abs(Zxx)
        magnitude_db = 20 * np.log10(magnitude + 1e-10)

        features = {}

        # Variação temporal do espectro
        spectral_flux = np.diff(magnitude, axis=1)
        max_flux = np.max(spectral_flux)
        mean_flux = np.mean(spectral_flux)

        features['spectral_flux_max'] = float(max_flux)
        features['spectral_flux_mean'] = float(mean_flux)

        # Detectar transientes (mudanças abruptas)
        transient_detected = max_flux > self.transient_threshold

        # Centróide espectral ao longo do tempo
        spectral_centroid = np.sum(f[:, np.newaxis] * magnitude, axis=0) / (np.sum(magnitude, axis=0) + 1e-10)
        centroid_variance = np.var(spectral_centroid)

        features['spectral_centroid_variance'] = float(centroid_variance)

        # Bandwidth espectral
        spectral_spread = np.sqrt(
            np.sum(((f[:, np.newaxis] - spectral_centroid) ** 2) * magnitude, axis=0) /
            (np.sum(magnitude, axis=0) + 1e-10)
        )
        spread_mean = np.mean(spectral_spread)

        features['spectral_spread_mean'] = float(spread_mean)

        # Anomalia: transiente forte + alta variância
        anomaly_detected = transient_detected or (centroid_variance > 1e6)

        confidence = 0.0
        if transient_detected:
            confidence += 0.6
        if centroid_variance > 1e6:
            confidence += 0.4

        confidence = min(confidence, 1.0)

        return AudioAnalysisResult(
            variant_name=self.name,
            anomaly_detected=anomaly_detected,
            confidence=confidence,
            features=features,
            metadata={'num_frames': magnitude.shape[1]}
        )


# ============================================================================
# VARIANTE 3: Wavelet Transform - Análise Multi-Resolução
# ============================================================================

class WaveletVariant(AudioVariant):
    """
    Transformada Wavelet para análise multi-resolução

    Detecta padrões em diferentes escalas temporais
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        wavelet: str = 'db4',
        level: int = 6,
        energy_threshold: float = 0.1
    ):
        super().__init__(sample_rate)
        self.name = "wavelet_multiresolution"
        self.wavelet = wavelet
        self.level = level
        self.energy_threshold = energy_threshold

    def analyze(self, audio: np.ndarray) -> AudioAnalysisResult:
        """Análise Wavelet"""

        import pywt

        # Decomposição wavelet
        coeffs = pywt.wavedec(audio, self.wavelet, level=self.level)

        features = {}

        # Energia em cada nível
        for i, coeff in enumerate(coeffs):
            energy = np.sum(coeff ** 2) / len(coeff)
            features[f'wavelet_energy_level_{i}'] = float(energy)

        # Detectar anomalia: energia concentrada em níveis específicos
        # Níveis altos = frequências baixas (infrassom potencial)
        # Níveis baixos = frequências altas (ultrassom potencial)

        high_level_energy = np.sum(coeffs[-1] ** 2)
        low_level_energy = np.sum(coeffs[0] ** 2)
        total_energy = sum(np.sum(c ** 2) for c in coeffs)

        high_level_ratio = high_level_energy / (total_energy + 1e-10)
        low_level_ratio = low_level_energy / (total_energy + 1e-10)

        features['high_level_energy_ratio'] = float(high_level_ratio)
        features['low_level_energy_ratio'] = float(low_level_ratio)

        # Anomalia: energia anormal em extremos
        anomaly_detected = (high_level_ratio > self.energy_threshold) or (low_level_ratio > 0.5)

        confidence = 0.0
        if high_level_ratio > self.energy_threshold:
            confidence += 0.5
        if low_level_ratio > 0.5:
            confidence += 0.5

        confidence = min(confidence, 1.0)

        return AudioAnalysisResult(
            variant_name=self.name,
            anomaly_detected=anomaly_detected,
            confidence=confidence,
            features=features,
            metadata={'levels': self.level}
        )


# ============================================================================
# VARIANTE 4: Análise de Formantes (EVP Detection)
# ============================================================================

class FormantVariant(AudioVariant):
    """
    Análise de formantes para detecção de voz/EVP

    Usa LPC (Linear Predictive Coding) para detectar estrutura de fala
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        lpc_order: int = 12,
        min_formants: int = 2
    ):
        super().__init__(sample_rate)
        self.name = "formant_evp"
        self.lpc_order = lpc_order
        self.min_formants = min_formants

    def analyze(self, audio: np.ndarray) -> AudioAnalysisResult:
        """Análise de formantes"""

        try:
            # LPC
            lpc_coeffs = librosa.lpc(audio, order=self.lpc_order)

            # Encontrar raízes
            roots = np.roots(lpc_coeffs)

            # Converter para frequências
            angles = np.angle(roots)
            frequencies = angles * (self.sample_rate / (2 * np.pi))

            # Filtrar formantes válidos (voz humana: 250-3800 Hz)
            formants = sorted([f for f in frequencies if 250 < f < 3800])

            features = {}

            if len(formants) >= 2:
                features['formant_f1'] = float(formants[0])
                features['formant_f2'] = float(formants[1])

                if len(formants) >= 3:
                    features['formant_f3'] = float(formants[2])

                # Verificar se formantes estão na faixa de voz
                f1_valid = 250 < formants[0] < 1000
                f2_valid = 700 < formants[1] < 3800

                features['formant_f1_valid'] = f1_valid
                features['formant_f2_valid'] = f2_valid

                # Espaçamento entre formantes
                f1_f2_spacing = formants[1] - formants[0]
                features['f1_f2_spacing'] = float(f1_f2_spacing)

                # Anomalia: formantes de voz detectados
                voice_like = f1_valid and f2_valid and (600 < f1_f2_spacing < 1400)

                anomaly_detected = voice_like
                confidence = 0.0

                if f1_valid:
                    confidence += 0.3
                if f2_valid:
                    confidence += 0.3
                if voice_like:
                    confidence += 0.4

            else:
                features['formant_count'] = len(formants)
                anomaly_detected = False
                confidence = 0.0

        except Exception as e:
            features = {'error': str(e)}
            anomaly_detected = False
            confidence = 0.0

        return AudioAnalysisResult(
            variant_name=self.name,
            anomaly_detected=anomaly_detected,
            confidence=confidence,
            features=features,
            metadata={'lpc_order': self.lpc_order}
        )


# ============================================================================
# VARIANTE 5: Análise de Energia em Sub-bandas (Filterbank)
# ============================================================================

class FilterbankVariant(AudioVariant):
    """
    Análise de energia em múltiplas sub-bandas (filterbank)

    Divide espectro em bandas e analisa energia em cada uma
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        num_bands: int = 32,
        anomaly_threshold_db: float = -30
    ):
        super().__init__(sample_rate)
        self.name = "filterbank_energy"
        self.num_bands = num_bands
        self.anomaly_threshold = anomaly_threshold_db

    def analyze(self, audio: np.ndarray) -> AudioAnalysisResult:
        """Análise filterbank"""

        # Mel-spectrogram (filterbank mel)
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.num_bands,
            fmax=self.sample_rate // 2
        )

        # Converter para dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        features = {}

        # Energia por banda
        band_energies = np.mean(mel_spec_db, axis=1)

        for i, energy in enumerate(band_energies):
            features[f'band_{i}_energy_db'] = float(energy)

        # Detectar bandas anômalas (energia muito alta)
        max_band_energy = np.max(band_energies)
        max_band_idx = np.argmax(band_energies)

        features['max_band_energy_db'] = float(max_band_energy)
        features['max_band_index'] = int(max_band_idx)

        # Entropia espectral (distribuição de energia)
        normalized_energy = band_energies - np.min(band_energies)
        normalized_energy = normalized_energy / (np.sum(normalized_energy) + 1e-10)
        spectral_entropy = -np.sum(normalized_energy * np.log2(normalized_energy + 1e-10))

        features['spectral_entropy'] = float(spectral_entropy)

        # Anomalia: energia concentrada em poucas bandas (baixa entropia)
        # ou energia muito alta em alguma banda
        low_entropy = spectral_entropy < 3.0
        high_energy_band = max_band_energy > self.anomaly_threshold

        anomaly_detected = low_entropy or high_energy_band

        confidence = 0.0
        if low_entropy:
            confidence += 0.4
        if high_energy_band:
            confidence += 0.6

        confidence = min(confidence, 1.0)

        return AudioAnalysisResult(
            variant_name=self.name,
            anomaly_detected=anomaly_detected,
            confidence=confidence,
            features=features,
            metadata={'num_bands': self.num_bands}
        )


# ============================================================================
# VARIANTE 6: Zero Crossing Rate - Análise de Periodicidade
# ============================================================================

class ZeroCrossingVariant(AudioVariant):
    """
    Análise de Zero Crossing Rate (ZCR)

    Detecta periodicidade e ruído vs. sinais tonais
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        frame_length: int = 2048,
        hop_length: int = 512
    ):
        super().__init__(sample_rate)
        self.name = "zero_crossing"
        self.frame_length = frame_length
        self.hop_length = hop_length

    def analyze(self, audio: np.ndarray) -> AudioAnalysisResult:
        """Análise ZCR"""

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]

        features = {}

        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        zcr_max = np.max(zcr)

        features['zcr_mean'] = float(zcr_mean)
        features['zcr_std'] = float(zcr_std)
        features['zcr_max'] = float(zcr_max)

        # ZCR alto = ruído branco ou frequências altas
        # ZCR baixo = tom puro ou frequências baixas
        # ZCR variável = fala

        high_zcr = zcr_mean > 0.15
        variable_zcr = zcr_std > 0.05

        features['high_zcr'] = high_zcr
        features['variable_zcr'] = variable_zcr

        # Anomalia: ZCR anormalmente alto ou muito variável
        anomaly_detected = high_zcr or variable_zcr

        confidence = 0.0
        if high_zcr:
            confidence += 0.5
        if variable_zcr:
            confidence += 0.5

        confidence = min(confidence, 1.0)

        return AudioAnalysisResult(
            variant_name=self.name,
            anomaly_detected=anomaly_detected,
            confidence=confidence,
            features=features,
            metadata={}
        )


# ============================================================================
# ENSEMBLE: Combinar Múltiplas Variantes
# ============================================================================

class AudioEnsemble:
    """
    Combina múltiplas variantes de análise de áudio

    Usa votação ponderada para decisão final
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        variants: Optional[List[AudioVariant]] = None,
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            sample_rate: Taxa de amostragem
            variants: Lista de variantes a usar (None = todas)
            weights: Pesos para cada variante (None = uniforme)
        """
        self.sample_rate = sample_rate

        # Se não especificado, usar todas as variantes
        if variants is None:
            self.variants = [
                FFTVariant(sample_rate),
                STFTVariant(sample_rate),
                WaveletVariant(sample_rate),
                FormantVariant(sample_rate),
                FilterbankVariant(sample_rate),
                ZeroCrossingVariant(sample_rate)
            ]
        else:
            self.variants = variants

        # Pesos uniformes se não especificado
        if weights is None:
            self.weights = [1.0 / len(self.variants)] * len(self.variants)
        else:
            # Normalizar pesos
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def analyze(self, audio: np.ndarray) -> Dict:
        """
        Analisa áudio com todas as variantes e combina resultados

        Returns:
            Dict com resultados combinados
        """
        results = []

        # Executar cada variante
        for variant in self.variants:
            try:
                result = variant.analyze(audio)
                results.append(result)
            except Exception as e:
                print(f"Erro na variante {variant.name}: {e}")
                continue

        if not results:
            return {
                'anomaly_detected': False,
                'confidence': 0.0,
                'num_variants': 0
            }

        # Combinar resultados
        anomaly_votes = sum(1 for r in results if r.anomaly_detected)
        weighted_confidence = sum(
            r.confidence * w
            for r, w in zip(results, self.weights[:len(results)])
        )

        # Decisão final: maioria + confiança ponderada
        final_anomaly = (anomaly_votes >= len(results) / 2) or (weighted_confidence > 0.5)

        # Agregar features
        all_features = {}
        for result in results:
            for key, value in result.features.items():
                all_features[f"{result.variant_name}.{key}"] = value

        return {
            'anomaly_detected': final_anomaly,
            'confidence': weighted_confidence,
            'num_variants': len(results),
            'num_anomalies_detected': anomaly_votes,
            'individual_results': [
                {
                    'variant': r.variant_name,
                    'anomaly': r.anomaly_detected,
                    'confidence': r.confidence
                }
                for r in results
            ],
            'features': all_features
        }


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

def example_usage():
    """Exemplo de uso das variantes de áudio"""

    # Gerar áudio de teste (tom puro + ruído)
    duration = 5.0  # segundos
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Tom de 440 Hz (Lá)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Adicionar ruído
    audio += 0.1 * np.random.randn(len(audio))

    # Adicionar transiente (evento anômalo)
    audio[len(audio) // 2: len(audio) // 2 + 1000] += 0.8 * np.random.randn(1000)

    print("=" * 60)
    print("TESTE DE VARIANTES DE ÁUDIO")
    print("=" * 60)

    # 1. FFT Clássica
    print("\n1. FFT Clássica:")
    fft_variant = FFTVariant(sample_rate)
    result = fft_variant.analyze(audio)
    print(f"   Anomalia: {result.anomaly_detected}")
    print(f"   Confiança: {result.confidence:.2f}")
    print(f"   Peak Freq: {result.features['peak_frequency_hz']:.1f} Hz")

    # 2. STFT
    print("\n2. STFT (Temporal):")
    stft_variant = STFTVariant(sample_rate)
    result = stft_variant.analyze(audio)
    print(f"   Anomalia: {result.anomaly_detected}")
    print(f"   Confiança: {result.confidence:.2f}")
    print(f"   Flux Max: {result.features['spectral_flux_max']:.2f}")

    # 3. Ensemble (todas as variantes)
    print("\n3. Ensemble (Todas as variantes):")
    ensemble = AudioEnsemble(sample_rate)
    result = ensemble.analyze(audio)
    print(f"   Anomalia: {result['anomaly_detected']}")
    print(f"   Confiança: {result['confidence']:.2f}")
    print(f"   Variantes detectando: {result['num_anomalies_detected']}/{result['num_variants']}")

    print("\n   Resultados individuais:")
    for ind_result in result['individual_results']:
        print(f"   - {ind_result['variant']}: {ind_result['anomaly']} (conf: {ind_result['confidence']:.2f})")


if __name__ == "__main__":
    example_usage()
