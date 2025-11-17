/**
 * Algoritmos Avançados de Precisão - Cliente Android
 *
 * Implementa equações matemáticas sofisticadas para melhorar precisão
 */

package com.spectral.algorithms

import kotlin.math.*


// ============================================================================
// COMPLEMENTARY FILTER - Fusão de Acelerômetro + Giroscópio
// ============================================================================

/**
 * Filtro Complementar - Mais leve que Kalman, ótimo para tempo real
 *
 * Equação:
 * θ(t) = α * (θ(t-1) + ω * dt) + (1 - α) * θ_accel
 *
 * Onde:
 * - θ = ângulo estimado
 * - ω = velocidade angular do giroscópio
 * - θ_accel = ângulo do acelerômetro
 * - α = coeficiente (tipicamente 0.96-0.98)
 */
class ComplementaryFilter(
    private val alpha: Float = 0.98f
) {
    private var roll: Float = 0f
    private var pitch: Float = 0f
    private var yaw: Float = 0f

    data class Orientation(
        val roll: Float,
        val pitch: Float,
        val yaw: Float,
        val accuracy: Float  // Estimativa de incerteza
    )

    fun update(
        accelX: Float, accelY: Float, accelZ: Float,
        gyroX: Float, gyroY: Float, gyroZ: Float,
        dt: Float
    ): Orientation {
        // Integrar giroscópio (alta frequência, sem drift)
        val gyroRoll = roll + gyroX * dt
        val gyroPitch = pitch + gyroY * dt
        val gyroYaw = yaw + gyroZ * dt

        // Calcular ângulos do acelerômetro (baixa frequência, absoluto)
        val accelRoll = atan2(accelY, accelZ)
        val accelPitch = atan2(-accelX, sqrt(accelY * accelY + accelZ * accelZ))

        // Fusão complementar
        roll = alpha * gyroRoll + (1 - alpha) * accelRoll
        pitch = alpha * gyroPitch + (1 - alpha) * accelPitch
        yaw = gyroYaw  // Yaw não pode ser corrigido pelo acelerômetro

        // Normalizar ângulos para [-π, π]
        roll = normalizeAngle(roll)
        pitch = normalizeAngle(pitch)
        yaw = normalizeAngle(yaw)

        // Calcular incerteza baseado na divergência
        val divergence = abs(gyroRoll - accelRoll) + abs(gyroPitch - accelPitch)
        val accuracy = 1.0f / (1.0f + divergence)

        return Orientation(roll, pitch, yaw, accuracy)
    }

    private fun normalizeAngle(angle: Float): Float {
        var normalized = angle
        while (normalized > PI) normalized -= (2 * PI).toFloat()
        while (normalized < -PI) normalized += (2 * PI).toFloat()
        return normalized
    }
}


// ============================================================================
// ALLAN VARIANCE - Análise de Estabilidade do Sensor
// ============================================================================

/**
 * Allan Variance - Detecta drift e ruído do sensor ao longo do tempo
 *
 * Equação:
 * σ²(τ) = 1/(2τ²(N-1)) * Σ[(x_{i+1} - x_i)²]
 *
 * Onde:
 * - τ = tempo de integração
 * - N = número de amostras
 * - x_i = média sobre o intervalo i
 */
class AllanVariance(private val windowSize: Int = 1000) {

    private val samples = mutableListOf<Float>()

    data class VarianceResult(
        val variance: Double,
        val drift: Double,      // rad/s (para giroscópio)
        val noise: Double,      // Random walk
        val stability: Float    // [0-1], 1 = muito estável
    )

    fun addSample(value: Float) {
        samples.add(value)
        if (samples.size > windowSize) {
            samples.removeAt(0)
        }
    }

    fun calculate(tau: Int = 1): VarianceResult {
        if (samples.size < 2 * tau) {
            return VarianceResult(0.0, 0.0, 0.0, 1.0f)
        }

        // Calcular médias em janelas de τ
        val means = mutableListOf<Double>()
        for (i in 0 until samples.size - tau step tau) {
            val window = samples.subList(i, min(i + tau, samples.size))
            means.add(window.average())
        }

        // Calcular Allan Variance
        var sumSquaredDiff = 0.0
        for (i in 0 until means.size - 1) {
            val diff = means[i + 1] - means[i]
            sumSquaredDiff += diff * diff
        }

        val variance = sumSquaredDiff / (2.0 * means.size)

        // Estimar drift (inclinação da variância vs tau)
        val drift = sqrt(variance) / tau

        // Estimar ruído (random walk)
        val noise = sqrt(variance) * sqrt(tau.toDouble())

        // Calcular estabilidade (0 = instável, 1 = estável)
        val stability = (1.0 / (1.0 + variance)).toFloat()

        return VarianceResult(variance, drift, noise, stability)
    }
}


// ============================================================================
// OUTLIER DETECTION - Detecção de Valores Anômalos
// ============================================================================

/**
 * Detecção de Outliers usando MAD (Median Absolute Deviation)
 *
 * Equação:
 * MAD = median(|X_i - median(X)|)
 * Outlier se: |X_i - median(X)| > k * MAD
 *
 * Onde k = 3.5 (threshold padrão)
 */
class OutlierDetector(
    private val threshold: Float = 3.5f,
    private val windowSize: Int = 100
) {
    private val buffer = ArrayDeque<Float>(windowSize)

    data class OutlierResult(
        val isOutlier: Boolean,
        val zScore: Float,
        val confidence: Float
    )

    fun detect(value: Float): OutlierResult {
        buffer.addLast(value)
        if (buffer.size > windowSize) {
            buffer.removeFirst()
        }

        if (buffer.size < 10) {
            return OutlierResult(false, 0f, 1f)
        }

        val sorted = buffer.sorted()
        val median = sorted[sorted.size / 2]

        // Calcular MAD
        val deviations = buffer.map { abs(it - median) }.sorted()
        val mad = deviations[deviations.size / 2]

        if (mad < 1e-6f) {
            return OutlierResult(false, 0f, 1f)
        }

        // Z-score modificado
        val zScore = 0.6745f * abs(value - median) / mad

        val isOutlier = zScore > threshold
        val confidence = 1.0f / (1.0f + zScore / threshold)

        return OutlierResult(isOutlier, zScore, confidence)
    }
}


// ============================================================================
// SIGNAL QUALITY METRICS - Métricas de Qualidade do Sinal
// ============================================================================

/**
 * Calcula múltiplas métricas de qualidade do sinal
 */
class SignalQualityAnalyzer(private val windowSize: Int = 100) {

    private val buffer = ArrayDeque<Float>(windowSize)

    data class QualityMetrics(
        val snr: Float,              // Signal-to-Noise Ratio (dB)
        val entropy: Float,          // Shannon Entropy
        val kurtosis: Float,         // Kurtose (picos)
        val skewness: Float,         // Assimetria
        val overallQuality: Float    // [0-1]
    )

    fun analyze(value: Float): QualityMetrics {
        buffer.addLast(value)
        if (buffer.size > windowSize) {
            buffer.removeFirst()
        }

        if (buffer.size < 10) {
            return QualityMetrics(0f, 0f, 0f, 0f, 0f)
        }

        val mean = buffer.average().toFloat()
        val variance = buffer.map { (it - mean) * (it - mean) }.average().toFloat()
        val std = sqrt(variance)

        // SNR = 10 * log10(signal_power / noise_power)
        val signalPower = mean * mean
        val noisePower = variance
        val snr = if (noisePower > 1e-6f) {
            10 * log10(signalPower / noisePower)
        } else {
            100f  // SNR muito alto
        }

        // Shannon Entropy
        val entropy = calculateEntropy(buffer.toList())

        // Kurtose (4º momento, detecta picos)
        val kurtosis = buffer.map { ((it - mean) / std).pow(4) }.average().toFloat() - 3f

        // Skewness (3º momento, detecta assimetria)
        val skewness = buffer.map { ((it - mean) / std).pow(3) }.average().toFloat()

        // Qualidade geral [0-1]
        val snrQuality = (snr / 40f).coerceIn(0f, 1f)  // SNR de 40dB = perfeito
        val entropyQuality = 1.0f / (1.0f + abs(entropy - 0.5f))
        val kurtosisQuality = 1.0f / (1.0f + abs(kurtosis))

        val overallQuality = (snrQuality + entropyQuality + kurtosisQuality) / 3f

        return QualityMetrics(snr, entropy, kurtosis, skewness, overallQuality)
    }

    private fun calculateEntropy(data: List<Float>): Float {
        // Discretizar em bins
        val numBins = 20
        val min = data.minOrNull() ?: 0f
        val max = data.maxOrNull() ?: 1f
        val binSize = (max - min) / numBins

        if (binSize < 1e-6f) return 0f

        val histogram = IntArray(numBins)
        for (value in data) {
            val bin = ((value - min) / binSize).toInt().coerceIn(0, numBins - 1)
            histogram[bin]++
        }

        // Shannon Entropy: H = -Σ p(x) * log2(p(x))
        var entropy = 0.0
        for (count in histogram) {
            if (count > 0) {
                val p = count.toDouble() / data.size
                entropy -= p * log2(p)
            }
        }

        // Normalizar para [0, 1]
        return (entropy / log2(numBins.toDouble())).toFloat()
    }
}


// ============================================================================
// ADAPTIVE THRESHOLD - Limiar Adaptativo Inteligente
// ============================================================================

/**
 * Calcula limiar adaptativo usando algoritmo CUSUM (Cumulative Sum)
 *
 * Equação:
 * S_i = max(0, S_{i-1} + (x_i - μ - k))
 *
 * Detecta mudança se S_i > h (threshold)
 */
class AdaptiveThreshold(
    private val k: Float = 0.5f,    // Slack parameter
    private val h: Float = 5.0f,    // Threshold
    private val alpha: Float = 0.05f // Learning rate
) {
    private var mean: Float = 0f
    private var std: Float = 1f
    private var cumulativeSum: Float = 0f
    private var sampleCount: Int = 0

    data class ThresholdResult(
        val threshold: Float,
        val isAnomaly: Boolean,
        val cumulativeSum: Float,
        val zScore: Float
    )

    fun update(value: Float): ThresholdResult {
        // Atualizar média e std incrementalmente (Welford's algorithm)
        sampleCount++
        val delta = value - mean
        mean += delta / sampleCount

        val delta2 = value - mean
        val variance = if (sampleCount > 1) {
            ((sampleCount - 2) * std * std + delta * delta2) / (sampleCount - 1)
        } else {
            1.0f
        }
        std = sqrt(variance)

        // CUSUM
        val z = (value - mean) / (std + 1e-6f)
        cumulativeSum = max(0f, cumulativeSum + (z - k))

        val isAnomaly = cumulativeSum > h

        // Se detectou mudança, resetar
        if (isAnomaly) {
            cumulativeSum = 0f
        }

        // Limiar adaptativo
        val threshold = mean + h * std

        return ThresholdResult(threshold, isAnomaly, cumulativeSum, z)
    }
}


// ============================================================================
// UNCERTAINTY QUANTIFICATION - Quantificação de Incerteza
// ============================================================================

/**
 * Propaga incerteza através de múltiplas medições
 *
 * Equação (Lei de Propagação de Incerteza):
 * σ_f² = Σ (∂f/∂x_i)² * σ_x_i²
 */
class UncertaintyPropagation {

    data class Measurement(
        val value: Float,
        val uncertainty: Float
    )

    /**
     * Magnitude de vetor com propagação de incerteza
     * f(x,y,z) = sqrt(x² + y² + z²)
     */
    fun vectorMagnitude(
        x: Measurement,
        y: Measurement,
        z: Measurement
    ): Measurement {
        val magnitude = sqrt(x.value * x.value + y.value * y.value + z.value * z.value)

        // Derivadas parciais
        val dfdx = x.value / (magnitude + 1e-6f)
        val dfdy = y.value / (magnitude + 1e-6f)
        val dfdz = z.value / (magnitude + 1e-6f)

        // Propagação de incerteza
        val uncertaintySquared =
            dfdx * dfdx * x.uncertainty * x.uncertainty +
            dfdy * dfdy * y.uncertainty * y.uncertainty +
            dfdz * dfdz * z.uncertainty * z.uncertainty

        val uncertainty = sqrt(uncertaintySquared)

        return Measurement(magnitude, uncertainty)
    }

    /**
     * Média ponderada por incerteza
     */
    fun weightedAverage(measurements: List<Measurement>): Measurement {
        if (measurements.isEmpty()) {
            return Measurement(0f, Float.MAX_VALUE)
        }

        // Pesos = 1 / σ²
        val weights = measurements.map { 1.0f / (it.uncertainty * it.uncertainty + 1e-6f) }
        val totalWeight = weights.sum()

        val weightedSum = measurements.zip(weights).sumOf {
            (it.first.value * it.second).toDouble()
        }.toFloat()

        val average = weightedSum / totalWeight

        // Incerteza da média ponderada
        val uncertainty = sqrt(1.0f / totalWeight)

        return Measurement(average, uncertainty)
    }
}


// ============================================================================
// SENSOR FUSION AVANÇADO
// ============================================================================

/**
 * Fusão avançada com múltiplos sensores e pesos adaptativos
 */
class AdvancedSensorFusion {

    data class SensorData(
        val value: Float,
        val quality: Float,      // [0-1]
        val timestamp: Long
    )

    data class FusionResult(
        val value: Float,
        val confidence: Float,
        val weights: Map<String, Float>
    )

    /**
     * Fusão adaptativa baseada em qualidade
     *
     * Equação:
     * x_fused = Σ (w_i * x_i) / Σ w_i
     *
     * Onde w_i = quality_i² (peso quadrático para enfatizar alta qualidade)
     */
    fun fuse(sensors: Map<String, SensorData>): FusionResult {
        if (sensors.isEmpty()) {
            return FusionResult(0f, 0f, emptyMap())
        }

        // Calcular pesos baseados em qualidade (quadrático)
        val weights = sensors.mapValues { (_, data) ->
            data.quality * data.quality
        }

        val totalWeight = weights.values.sum()

        if (totalWeight < 1e-6f) {
            return FusionResult(0f, 0f, weights)
        }

        // Média ponderada
        val fusedValue = sensors.map { (key, data) ->
            data.value * weights[key]!!
        }.sum() / totalWeight

        // Confiança = média geométrica das qualidades
        val confidence = sensors.values
            .map { it.quality.toDouble() }
            .reduce { acc, q -> acc * q }
            .pow(1.0 / sensors.size)
            .toFloat()

        // Normalizar pesos para soma = 1
        val normalizedWeights = weights.mapValues { it.value / totalWeight }

        return FusionResult(fusedValue, confidence, normalizedWeights)
    }
}
