package com.spectral.app.ui

import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import androidx.viewpager2.adapter.FragmentStateAdapter
import androidx.viewpager2.widget.ViewPager2
import com.google.android.material.tabs.TabLayout
import com.google.android.material.tabs.TabLayoutMediator
import com.spectral.app.R
import com.spectral.app.network.WebSocketClient
import com.spectral.app.processing.AdvancedAlgorithms.*
import kotlinx.coroutines.*
import org.json.JSONObject

/**
 * DetectionActivity - Tela de detecção em tempo real
 *
 * Features:
 * - Visualização de sensores em tempo real
 * - Detecção de anomalias
 * - WebSocket para servidor
 * - Algoritmos avançados de precisão
 */
class DetectionActivity : AppCompatActivity(), SensorEventListener {

    // Sensor Manager
    private lateinit var sensorManager: SensorManager

    // Sensores
    private var accelerometer: Sensor? = null
    private var gyroscope: Sensor? = null
    private var magnetometer: Sensor? = null

    // WebSocket
    private lateinit var websocketClient: WebSocketClient

    // Algoritmos avançados
    private lateinit var complementaryFilter: ComplementaryFilter
    private lateinit var allanVariance: AllanVariance
    private lateinit var signalQualityAnalyzer: SignalQualityAnalyzer
    private lateinit var advancedFusion: AdvancedSensorFusion

    // Views
    private lateinit var viewPager: ViewPager2
    private lateinit var tabLayout: TabLayout
    private lateinit var tvStatus: TextView
    private lateinit var tvAnomalyScore: TextView
    private lateinit var viewAnomalyIndicator: View

    // Coroutine scope
    private val scope = CoroutineScope(Dispatchers.Main + Job())

    // Estado
    private var isDetecting = false
    private var currentAnomalyScore = 0.0f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_detection)

        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        supportActionBar?.title = "Detecção em Tempo Real"

        // Inicializar
        initSensors()
        initAlgorithms()
        initViews()
        initWebSocket()

        // Iniciar detecção
        startDetection()
    }

    private fun initSensors() {
        sensorManager = getSystemService(SENSOR_SERVICE) as SensorManager

        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
    }

    private fun initAlgorithms() {
        complementaryFilter = ComplementaryFilter(alpha = 0.98f)
        allanVariance = AllanVariance(windowSize = 1000)
        signalQualityAnalyzer = SignalQualityAnalyzer(windowSize = 100)
        advancedFusion = AdvancedSensorFusion()
    }

    private fun initViews() {
        viewPager = findViewById(R.id.view_pager)
        tabLayout = findViewById(R.id.tab_layout)
        tvStatus = findViewById(R.id.tv_status)
        tvAnomalyScore = findViewById(R.id.tv_anomaly_score)
        viewAnomalyIndicator = findViewById(R.id.view_anomaly_indicator)

        // ViewPager com fragmentos
        viewPager.adapter = DetectionPagerAdapter(this)

        // Tabs
        TabLayoutMediator(tabLayout, viewPager) { tab, position ->
            tab.text = when (position) {
                0 -> "Sensores"
                1 -> "Espectro"
                2 -> "Câmera"
                else -> "Tab $position"
            }
        }.attach()
    }

    private fun initWebSocket() {
        val serverUrl = getServerUrl()
        val clientId = getClientId()

        websocketClient = WebSocketClient(
            url = "$serverUrl/ws/$clientId",
            onMessage = { message ->
                handleServerMessage(message)
            },
            onError = { error ->
                runOnUiThread {
                    tvStatus.text = "Erro: $error"
                }
            }
        )

        websocketClient.connect()
    }

    private fun startDetection() {
        isDetecting = true

        // Registrar listeners
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }

        gyroscope?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }

        magnetometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }

        tvStatus.text = "Detectando..."
    }

    private fun stopDetection() {
        isDetecting = false
        sensorManager.unregisterListener(this)
        tvStatus.text = "Parado"
    }

    // Dados temporários
    private var lastAccel = FloatArray(3)
    private var lastGyro = FloatArray(3)
    private var lastMag = FloatArray(3)

    override fun onSensorChanged(event: SensorEvent?) {
        event ?: return

        when (event.sensor.type) {
            Sensor.TYPE_ACCELEROMETER -> {
                lastAccel = event.values.clone()
                processAccelerometer(event.values)
            }
            Sensor.TYPE_GYROSCOPE -> {
                lastGyro = event.values.clone()
                processGyroscope(event.values)
            }
            Sensor.TYPE_MAGNETIC_FIELD -> {
                lastMag = event.values.clone()
                processMagnetometer(event.values)
            }
        }

        // Processar fusão
        processSensorFusion()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Handle accuracy changes
    }

    private fun processAccelerometer(values: FloatArray) {
        // Analisar qualidade
        val qualityX = signalQualityAnalyzer.analyze(values[0])

        // Enviar para fragmento de visualização
        sendToVisualization("accelerometer", values)
    }

    private fun processGyroscope(values: FloatArray) {
        // Analisar qualidade
        val qualityX = signalQualityAnalyzer.analyze(values[0])

        sendToVisualization("gyroscope", values)
    }

    private fun processMagnetometer(values: FloatArray) {
        // Allan variance para estabilidade
        val variance = allanVariance.add(values[0])

        // Análise de qualidade
        val quality = signalQualityAnalyzer.analyze(values[0])

        sendToVisualization("magnetometer", values)
    }

    private fun processSensorFusion() {
        // Complementary filter
        val dt = 0.02f  // ~50Hz
        val orientation = complementaryFilter.update(
            lastAccel[0], lastAccel[1], lastAccel[2],
            lastGyro[0], lastGyro[1], lastGyro[2],
            dt
        )

        // Advanced fusion
        val fusedData = advancedFusion.fuse(
            mapOf(
                "accelerometer" to lastAccel[0],
                "gyroscope" to lastGyro[0],
                "magnetometer" to lastMag[0]
            ),
            mapOf(
                "accelerometer" to 0.8f,
                "gyroscope" to 0.9f,
                "magnetometer" to 0.7f
            )
        )

        // Enviar para servidor
        sendSensorPacket(orientation.accuracy, fusedData.confidence)
    }

    private fun sendSensorPacket(orientationAccuracy: Float, fusionConfidence: Float) {
        scope.launch(Dispatchers.IO) {
            val packet = JSONObject().apply {
                put("timestamp", System.currentTimeMillis() * 1_000_000)  // Nanoseconds
                put("sensors", JSONObject().apply {
                    put("accelerometer", JSONObject().apply {
                        put("x", lastAccel[0])
                        put("y", lastAccel[1])
                        put("z", lastAccel[2])
                    })
                    put("gyroscope", JSONObject().apply {
                        put("x", lastGyro[0])
                        put("y", lastGyro[1])
                        put("z", lastGyro[2])
                    })
                    put("magnetometer", JSONObject().apply {
                        put("x", lastMag[0])
                        put("y", lastMag[1])
                        put("z", lastMag[2])
                        put("magnitude", kotlin.math.sqrt(
                            lastMag[0] * lastMag[0] +
                            lastMag[1] * lastMag[1] +
                            lastMag[2] * lastMag[2]
                        ))
                    })
                })
                put("quality", JSONObject().apply {
                    put("orientation_accuracy", orientationAccuracy)
                    put("fusion_confidence", fusionConfidence)
                })
            }

            websocketClient.send(packet.toString())
        }
    }

    private fun handleServerMessage(message: String) {
        try {
            val json = JSONObject(message)
            val type = json.getString("type")

            when (type) {
                "processing_result" -> {
                    val anomaly = json.getJSONObject("anomaly")
                    val detected = anomaly.getBoolean("detected")
                    val score = anomaly.getDouble("score").toFloat()
                    val confidence = anomaly.getDouble("confidence").toFloat()

                    updateAnomalyUI(detected, score, confidence)
                }
                "error" -> {
                    val errorMsg = json.getString("message")
                    runOnUiThread {
                        tvStatus.text = "Erro: $errorMsg"
                    }
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun updateAnomalyUI(detected: Boolean, score: Float, confidence: Float) {
        runOnUiThread {
            currentAnomalyScore = score

            tvAnomalyScore.text = String.format("Score: %.2f (Confiança: %.2f)", score, confidence)

            // Indicador visual
            val color = when {
                !detected -> android.graphics.Color.GREEN
                score < 0.5f -> android.graphics.Color.YELLOW
                else -> android.graphics.Color.RED
            }

            viewAnomalyIndicator.setBackgroundColor(color)

            // Vibrar se anomalia forte
            if (detected && score > 0.7f && confidence > 0.8f) {
                vibrate()
            }
        }
    }

    private fun vibrate() {
        val vibrator = getSystemService(VIBRATOR_SERVICE) as android.os.Vibrator
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
            vibrator.vibrate(
                android.os.VibrationEffect.createOneShot(
                    200,
                    android.os.VibrationEffect.DEFAULT_AMPLITUDE
                )
            )
        } else {
            @Suppress("DEPRECATION")
            vibrator.vibrate(200)
        }
    }

    private fun sendToVisualization(sensorType: String, values: FloatArray) {
        // Broadcast para fragmento de visualização
        val intent = android.content.Intent("SENSOR_DATA")
        intent.putExtra("sensor_type", sensorType)
        intent.putExtra("values", values)
        androidx.localbroadcastmanager.content.LocalBroadcastManager
            .getInstance(this)
            .sendBroadcast(intent)
    }

    private fun getServerUrl(): String {
        val prefs = getSharedPreferences("spectral_prefs", MODE_PRIVATE)
        return prefs.getString("server_url", "ws://192.168.1.100:8000") ?: "ws://192.168.1.100:8000"
    }

    private fun getClientId(): String {
        val prefs = getSharedPreferences("spectral_prefs", MODE_PRIVATE)
        var clientId = prefs.getString("client_id", null)

        if (clientId == null) {
            clientId = "android_${System.currentTimeMillis()}"
            prefs.edit().putString("client_id", clientId).apply()
        }

        return clientId
    }

    override fun onDestroy() {
        super.onDestroy()
        stopDetection()
        websocketClient.disconnect()
        scope.cancel()
    }

    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
    }

    // ViewPager Adapter
    private inner class DetectionPagerAdapter(activity: AppCompatActivity) :
        FragmentStateAdapter(activity) {

        override fun getItemCount(): Int = 3

        override fun createFragment(position: Int): Fragment {
            return when (position) {
                0 -> SensorVisualizationFragment()
                1 -> AudioSpectrumFragment()
                2 -> CameraPreviewFragment()
                else -> SensorVisualizationFragment()
            }
        }
    }
}
