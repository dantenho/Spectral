/**
 * Sistema de Calibração de Sensores - Android
 *
 * Calibra magnetômetro, acelerômetro e giroscópio antes do uso
 */

package com.spectral.calibration

import android.content.Context
import android.content.SharedPreferences
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlin.math.abs
import kotlin.math.sqrt

// ============================================================================
// MODELOS DE DADOS
// ============================================================================

data class CalibrationData(
    val offsetX: Float = 0f,
    val offsetY: Float = 0f,
    val offsetZ: Float = 0f,
    val scaleX: Float = 1f,
    val scaleY: Float = 1f,
    val scaleZ: Float = 1f,
    val timestamp: Long = System.currentTimeMillis(),
    val isCalibrated: Boolean = false
)

enum class CalibrationStep {
    WELCOME,              // Tela de boas-vindas
    PREPARE_MAGNETOMETER, // Preparação para magnetômetro
    CALIBRATE_MAG_1,      // Mag: Face para cima
    CALIBRATE_MAG_2,      // Mag: Face para baixo
    CALIBRATE_MAG_3,      // Mag: Lado esquerdo
    CALIBRATE_MAG_4,      // Mag: Lado direito
    CALIBRATE_MAG_5,      // Mag: Topo
    CALIBRATE_MAG_6,      // Mag: Fundo
    MAG_FIGURE_8,         // Mag: Figura 8 no ar
    PREPARE_ACCELEROMETER,// Preparação para acelerômetro
    CALIBRATE_ACCEL,      // Accel: Superfície plana
    PREPARE_GYROSCOPE,    // Preparação para giroscópio
    CALIBRATE_GYRO,       // Gyro: Manter imóvel
    COMPLETED             // Calibração completa
}

data class CalibrationState(
    val currentStep: CalibrationStep = CalibrationStep.WELCOME,
    val progress: Float = 0f,  // 0.0 - 1.0
    val samplesCollected: Int = 0,
    val samplesNeeded: Int = 100,
    val currentReadingX: Float = 0f,
    val currentReadingY: Float = 0f,
    val currentReadingZ: Float = 0f,
    val currentMagnitude: Float = 0f,
    val message: String = "",
    val canProceed: Boolean = false,
    val magnetometerData: CalibrationData = CalibrationData(),
    val accelerometerData: CalibrationData = CalibrationData(),
    val gyroscopeData: CalibrationData = CalibrationData()
)

// ============================================================================
// CALIBRATION MANAGER
// ============================================================================

class CalibrationManager(
    private val context: Context
) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val prefs: SharedPreferences = context.getSharedPreferences("calibration", Context.MODE_PRIVATE)

    // Sensores
    private val magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    // Estado da calibração
    private val _state = MutableStateFlow(CalibrationState())
    val state: StateFlow<CalibrationState> = _state.asStateFlow()

    // Buffers para coleta de dados
    private val magnetometerSamples = mutableListOf<FloatArray>()
    private val accelerometerSamples = mutableListOf<FloatArray>()
    private val gyroscopeSamples = mutableListOf<FloatArray>()

    private var currentSensor: Sensor? = null
    private var currentStepSamples = 0
    private val samplesPerStep = 100

    // Máximos e mínimos para calibração de magnetômetro
    private var magMaxX = Float.MIN_VALUE
    private var magMaxY = Float.MIN_VALUE
    private var magMaxZ = Float.MIN_VALUE
    private var magMinX = Float.MAX_VALUE
    private var magMinY = Float.MAX_VALUE
    private var magMinZ = Float.MAX_VALUE

    // ========================================================================
    // CONTROLE DE FLUXO
    // ========================================================================

    fun startCalibration() {
        loadExistingCalibration()
        _state.value = CalibrationState(
            currentStep = CalibrationStep.WELCOME,
            message = "Bem-vindo à calibração de sensores!\n\nEste processo levará cerca de 3 minutos."
        )
    }

    fun nextStep() {
        val currentStep = _state.value.currentStep
        val nextStep = when (currentStep) {
            CalibrationStep.WELCOME -> CalibrationStep.PREPARE_MAGNETOMETER
            CalibrationStep.PREPARE_MAGNETOMETER -> CalibrationStep.CALIBRATE_MAG_1
            CalibrationStep.CALIBRATE_MAG_1 -> CalibrationStep.CALIBRATE_MAG_2
            CalibrationStep.CALIBRATE_MAG_2 -> CalibrationStep.CALIBRATE_MAG_3
            CalibrationStep.CALIBRATE_MAG_3 -> CalibrationStep.CALIBRATE_MAG_4
            CalibrationStep.CALIBRATE_MAG_4 -> CalibrationStep.CALIBRATE_MAG_5
            CalibrationStep.CALIBRATE_MAG_5 -> CalibrationStep.CALIBRATE_MAG_6
            CalibrationStep.CALIBRATE_MAG_6 -> CalibrationStep.MAG_FIGURE_8
            CalibrationStep.MAG_FIGURE_8 -> {
                calculateMagnetometerCalibration()
                CalibrationStep.PREPARE_ACCELEROMETER
            }
            CalibrationStep.PREPARE_ACCELEROMETER -> CalibrationStep.CALIBRATE_ACCEL
            CalibrationStep.CALIBRATE_ACCEL -> {
                calculateAccelerometerCalibration()
                CalibrationStep.PREPARE_GYROSCOPE
            }
            CalibrationStep.PREPARE_GYROSCOPE -> CalibrationStep.CALIBRATE_GYRO
            CalibrationStep.CALIBRATE_GYRO -> {
                calculateGyroscopeCalibration()
                CalibrationStep.COMPLETED
            }
            CalibrationStep.COMPLETED -> CalibrationStep.COMPLETED
        }

        goToStep(nextStep)
    }

    private fun goToStep(step: CalibrationStep) {
        stopSensorListening()
        currentStepSamples = 0

        val message = when (step) {
            CalibrationStep.WELCOME -> "Bem-vindo à calibração!"
            CalibrationStep.PREPARE_MAGNETOMETER ->
                "🧲 CALIBRAÇÃO DO MAGNETÔMETRO\n\n" +
                "O magnetômetro detecta campos magnéticos.\n\n" +
                "Você precisará:\n" +
                "• Mover o celular em 6 orientações diferentes\n" +
                "• Fazer um movimento de figura 8 no ar\n\n" +
                "⚠️ Afaste-se de objetos metálicos!"

            CalibrationStep.CALIBRATE_MAG_1 ->
                "PASSO 1/6: FACE PARA CIMA\n\n" +
                "📱 Coloque o celular com a tela voltada para cima\n" +
                "em uma superfície plana.\n\n" +
                "Mantenha imóvel..."

            CalibrationStep.CALIBRATE_MAG_2 ->
                "PASSO 2/6: FACE PARA BAIXO\n\n" +
                "📱 Vire o celular com a tela voltada para baixo.\n\n" +
                "Mantenha imóvel..."

            CalibrationStep.CALIBRATE_MAG_3 ->
                "PASSO 3/6: LADO ESQUERDO\n\n" +
                "📱 Coloque o celular apoiado no lado esquerdo.\n\n" +
                "Mantenha imóvel..."

            CalibrationStep.CALIBRATE_MAG_4 ->
                "PASSO 4/6: LADO DIREITO\n\n" +
                "📱 Coloque o celular apoiado no lado direito.\n\n" +
                "Mantenha imóvel..."

            CalibrationStep.CALIBRATE_MAG_5 ->
                "PASSO 5/6: TOPO\n\n" +
                "📱 Coloque o celular apoiado no topo\n" +
                "(onde fica a câmera).\n\n" +
                "Mantenha imóvel..."

            CalibrationStep.CALIBRATE_MAG_6 ->
                "PASSO 6/6: FUNDO\n\n" +
                "📱 Coloque o celular apoiado no fundo\n" +
                "(onde fica a porta USB).\n\n" +
                "Mantenha imóvel..."

            CalibrationStep.MAG_FIGURE_8 ->
                "FIGURA 8 NO AR\n\n" +
                "✋ Segure o celular e faça movimentos\n" +
                "de figura 8 (∞) no ar.\n\n" +
                "Continue fazendo o movimento..."

            CalibrationStep.PREPARE_ACCELEROMETER ->
                "⚡ CALIBRAÇÃO DO ACELERÔMETRO\n\n" +
                "O acelerômetro mede aceleração e gravidade.\n\n" +
                "Você precisará:\n" +
                "• Colocar o celular em uma superfície plana\n" +
                "• Manter completamente imóvel\n\n" +
                "Preparado?"

            CalibrationStep.CALIBRATE_ACCEL ->
                "CALIBRANDO ACELERÔMETRO\n\n" +
                "📱 Coloque o celular em uma superfície\n" +
                "completamente plana e nivelada.\n\n" +
                "NÃO TOQUE no celular..."

            CalibrationStep.PREPARE_GYROSCOPE ->
                "🔄 CALIBRAÇÃO DO GIROSCÓPIO\n\n" +
                "O giroscópio mede rotação.\n\n" +
                "Você precisará:\n" +
                "• Manter o celular completamente imóvel\n" +
                "• Não tocá-lo durante a calibração\n\n" +
                "Preparado?"

            CalibrationStep.CALIBRATE_GYRO ->
                "CALIBRANDO GIROSCÓPIO\n\n" +
                "📱 Mantenha o celular COMPLETAMENTE IMÓVEL.\n\n" +
                "NÃO TOQUE no celular..."

            CalibrationStep.COMPLETED ->
                "✅ CALIBRAÇÃO COMPLETA!\n\n" +
                "Todos os sensores foram calibrados com sucesso.\n\n" +
                "Magnetômetro: ✓\n" +
                "Acelerômetro: ✓\n" +
                "Giroscópio: ✓"
        }

        _state.value = _state.value.copy(
            currentStep = step,
            progress = 0f,
            samplesCollected = 0,
            samplesNeeded = samplesPerStep,
            message = message,
            canProceed = (step == CalibrationStep.WELCOME ||
                          step == CalibrationStep.PREPARE_MAGNETOMETER ||
                          step == CalibrationStep.PREPARE_ACCELEROMETER ||
                          step == CalibrationStep.PREPARE_GYROSCOPE ||
                          step == CalibrationStep.COMPLETED)
        )

        // Iniciar coleta de dados se necessário
        when (step) {
            CalibrationStep.CALIBRATE_MAG_1,
            CalibrationStep.CALIBRATE_MAG_2,
            CalibrationStep.CALIBRATE_MAG_3,
            CalibrationStep.CALIBRATE_MAG_4,
            CalibrationStep.CALIBRATE_MAG_5,
            CalibrationStep.CALIBRATE_MAG_6,
            CalibrationStep.MAG_FIGURE_8 -> startMagnetometerCalibration()

            CalibrationStep.CALIBRATE_ACCEL -> startAccelerometerCalibration()
            CalibrationStep.CALIBRATE_GYRO -> startGyroscopeCalibration()
            else -> {}
        }
    }

    // ========================================================================
    // CALIBRAÇÃO DO MAGNETÔMETRO
    // ========================================================================

    private fun startMagnetometerCalibration() {
        currentSensor = magnetometer
        sensorManager.registerListener(this, magnetometer, SensorManager.SENSOR_DELAY_NORMAL)
    }

    private fun calculateMagnetometerCalibration() {
        // Calcular offsets (hard iron calibration)
        val offsetX = (magMaxX + magMinX) / 2f
        val offsetY = (magMaxY + magMinY) / 2f
        val offsetZ = (magMaxZ + magMinZ) / 2f

        // Calcular escala (soft iron calibration)
        val rangeX = magMaxX - magMinX
        val rangeY = magMaxY - magMinY
        val rangeZ = magMaxZ - magMinZ
        val avgRange = (rangeX + rangeY + rangeZ) / 3f

        val scaleX = if (abs(rangeX) > 1e-6) avgRange / rangeX else 1f
        val scaleY = if (abs(rangeY) > 1e-6) avgRange / rangeY else 1f
        val scaleZ = if (abs(rangeZ) > 1e-6) avgRange / rangeZ else 1f

        val calibration = CalibrationData(
            offsetX = offsetX,
            offsetY = offsetY,
            offsetZ = offsetZ,
            scaleX = scaleX,
            scaleY = scaleY,
            scaleZ = scaleZ,
            isCalibrated = true
        )

        _state.value = _state.value.copy(magnetometerData = calibration)

        // Salvar
        saveCalibration("magnetometer", calibration)
    }

    // ========================================================================
    // CALIBRAÇÃO DO ACELERÔMETRO
    // ========================================================================

    private fun startAccelerometerCalibration() {
        currentSensor = accelerometer
        accelerometerSamples.clear()
        sensorManager.registerListener(this, accelerometer, SensorManager.SENSOR_DELAY_NORMAL)
    }

    private fun calculateAccelerometerCalibration() {
        if (accelerometerSamples.isEmpty()) return

        // Calcular média (gravidade deve ser ~9.8 m/s² no eixo Z)
        val avgX = accelerometerSamples.map { it[0] }.average().toFloat()
        val avgY = accelerometerSamples.map { it[1] }.average().toFloat()
        val avgZ = accelerometerSamples.map { it[2] }.average().toFloat()

        // Offsets (assumindo superfície plana: X=0, Y=0, Z=9.8)
        val offsetX = avgX
        val offsetY = avgY
        val offsetZ = avgZ - 9.8f

        val calibration = CalibrationData(
            offsetX = offsetX,
            offsetY = offsetY,
            offsetZ = offsetZ,
            isCalibrated = true
        )

        _state.value = _state.value.copy(accelerometerData = calibration)

        saveCalibration("accelerometer", calibration)
    }

    // ========================================================================
    // CALIBRAÇÃO DO GIROSCÓPIO
    // ========================================================================

    private fun startGyroscopeCalibration() {
        currentSensor = gyroscope
        gyroscopeSamples.clear()
        sensorManager.registerListener(this, gyroscope, SensorManager.SENSOR_DELAY_NORMAL)
    }

    private fun calculateGyroscopeCalibration() {
        if (gyroscopeSamples.isEmpty()) return

        // Calcular drift (bias) médio
        val avgX = gyroscopeSamples.map { it[0] }.average().toFloat()
        val avgY = gyroscopeSamples.map { it[1] }.average().toFloat()
        val avgZ = gyroscopeSamples.map { it[2] }.average().toFloat()

        val calibration = CalibrationData(
            offsetX = avgX,
            offsetY = avgY,
            offsetZ = avgZ,
            isCalibrated = true
        )

        _state.value = _state.value.copy(gyroscopeData = calibration)

        saveCalibration("gyroscope", calibration)
    }

    // ========================================================================
    // SENSOR EVENT LISTENER
    // ========================================================================

    override fun onSensorChanged(event: SensorEvent) {
        val x = event.values[0]
        val y = event.values[1]
        val z = event.values[2]
        val magnitude = sqrt(x * x + y * y + z * z)

        // Atualizar leitura atual
        _state.value = _state.value.copy(
            currentReadingX = x,
            currentReadingY = y,
            currentReadingZ = z,
            currentMagnitude = magnitude
        )

        // Coletar amostras
        when (event.sensor.type) {
            Sensor.TYPE_MAGNETIC_FIELD -> {
                // Atualizar min/max para calibração hard/soft iron
                if (x > magMaxX) magMaxX = x
                if (y > magMaxY) magMaxY = y
                if (z > magMaxZ) magMaxZ = z
                if (x < magMinX) magMinX = x
                if (y < magMinY) magMinY = y
                if (z < magMinZ) magMinZ = z

                magnetometerSamples.add(floatArrayOf(x, y, z))
            }

            Sensor.TYPE_ACCELEROMETER -> {
                accelerometerSamples.add(floatArrayOf(x, y, z))
            }

            Sensor.TYPE_GYROSCOPE -> {
                gyroscopeSamples.add(floatArrayOf(x, y, z))
            }
        }

        currentStepSamples++

        // Atualizar progresso
        val progress = currentStepSamples.toFloat() / samplesPerStep
        _state.value = _state.value.copy(
            progress = progress.coerceIn(0f, 1f),
            samplesCollected = currentStepSamples
        )

        // Auto-avançar quando completo
        if (currentStepSamples >= samplesPerStep) {
            stopSensorListening()
            _state.value = _state.value.copy(canProceed = true)
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Não usado
    }

    // ========================================================================
    // PERSISTÊNCIA
    // ========================================================================

    private fun saveCalibration(sensorName: String, data: CalibrationData) {
        prefs.edit().apply {
            putFloat("${sensorName}_offset_x", data.offsetX)
            putFloat("${sensorName}_offset_y", data.offsetY)
            putFloat("${sensorName}_offset_z", data.offsetZ)
            putFloat("${sensorName}_scale_x", data.scaleX)
            putFloat("${sensorName}_scale_y", data.scaleY)
            putFloat("${sensorName}_scale_z", data.scaleZ)
            putLong("${sensorName}_timestamp", data.timestamp)
            putBoolean("${sensorName}_calibrated", true)
        }.apply()
    }

    private fun loadExistingCalibration() {
        val magData = loadCalibration("magnetometer")
        val accelData = loadCalibration("accelerometer")
        val gyroData = loadCalibration("gyroscope")

        _state.value = _state.value.copy(
            magnetometerData = magData,
            accelerometerData = accelData,
            gyroscopeData = gyroData
        )
    }

    private fun loadCalibration(sensorName: String): CalibrationData {
        return CalibrationData(
            offsetX = prefs.getFloat("${sensorName}_offset_x", 0f),
            offsetY = prefs.getFloat("${sensorName}_offset_y", 0f),
            offsetZ = prefs.getFloat("${sensorName}_offset_z", 0f),
            scaleX = prefs.getFloat("${sensorName}_scale_x", 1f),
            scaleY = prefs.getFloat("${sensorName}_scale_y", 1f),
            scaleZ = prefs.getFloat("${sensorName}_scale_z", 1f),
            timestamp = prefs.getLong("${sensorName}_timestamp", 0),
            isCalibrated = prefs.getBoolean("${sensorName}_calibrated", false)
        )
    }

    fun getCalibration(sensorName: String): CalibrationData {
        return loadCalibration(sensorName)
    }

    // ========================================================================
    // APLICAR CALIBRAÇÃO
    // ========================================================================

    fun applyCalibratedValues(
        sensorName: String,
        rawX: Float,
        rawY: Float,
        rawZ: Float
    ): FloatArray {
        val calibration = getCalibration(sensorName)

        if (!calibration.isCalibrated) {
            return floatArrayOf(rawX, rawY, rawZ)
        }

        // Aplicar offsets e escala
        val calibratedX = (rawX - calibration.offsetX) * calibration.scaleX
        val calibratedY = (rawY - calibration.offsetY) * calibration.scaleY
        val calibratedZ = (rawZ - calibration.offsetZ) * calibration.scaleZ

        return floatArrayOf(calibratedX, calibratedY, calibratedZ)
    }

    // ========================================================================
    // CLEANUP
    // ========================================================================

    private fun stopSensorListening() {
        sensorManager.unregisterListener(this)
        currentSensor = null
    }

    fun cleanup() {
        stopSensorListening()
    }
}
