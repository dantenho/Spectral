# 📱 Especificação Técnica - Cliente Android

## 🎯 Visão Geral

Cliente Android desenvolvido em Kotlin para o dispositivo **OPPO Reno 11 F5 (5G)** que coleta dados de múltiplos sensores, processa informações usando Edge AI (NPU) e transmite dados em tempo real para o servidor.

---

## 🔧 Stack Tecnológico

### Core
- **Linguagem**: Kotlin 1.9+
- **Min SDK**: Android 8.0 (API 26)
- **Target SDK**: Android 14 (API 34)
- **Build System**: Gradle 8.0+ com Kotlin DSL

### Bibliotecas Principais

```kotlin
dependencies {
    // Core Android
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")

    // Architecture Components
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.7.3")

    // Network - Ktor Client
    implementation("io.ktor:ktor-client-core:2.3.7")
    implementation("io.ktor:ktor-client-android:2.3.7")
    implementation("io.ktor:ktor-client-websockets:2.3.7")
    implementation("io.ktor:ktor-client-serialization:2.3.7")
    implementation("io.ktor:ktor-client-logging:2.3.7")

    // JSON Serialization
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.2")

    // Edge AI - TensorFlow Lite
    implementation("org.tensorflow:tensorflow-lite:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")

    // Camera
    implementation("androidx.camera:camera-core:1.3.1")
    implementation("androidx.camera:camera-camera2:1.3.1")
    implementation("androidx.camera:camera-lifecycle:1.3.1")
    implementation("androidx.camera:camera-video:1.3.1")

    // Audio Processing
    implementation("com.github.paramsen:noise:2.0.0")  // FFT

    // Permissions
    implementation("com.google.accompanist:accompanist-permissions:0.34.0")

    // Logging
    implementation("com.jakewharton.timber:timber:5.0.1")
}
```

---

## 🏗️ Arquitetura MVVM

```
app/
├── data/
│   ├── model/               # Data classes
│   │   ├── SensorData.kt
│   │   ├── MagneticData.kt
│   │   ├── AudioData.kt
│   │   ├── VideoFrame.kt
│   │   └── NetworkPacket.kt
│   ├── repository/          # Data sources
│   │   ├── SensorRepository.kt
│   │   └── NetworkRepository.kt
│   └── remote/              # Network layer
│       └── WebSocketClient.kt
├── domain/
│   ├── usecase/             # Business logic
│   │   ├── CollectSensorDataUseCase.kt
│   │   ├── ProcessEdgeAIUseCase.kt
│   │   └── TransmitDataUseCase.kt
│   └── mapper/              # Data transformations
│       └── SensorDataMapper.kt
├── presentation/
│   ├── ui/
│   │   ├── main/
│   │   │   ├── MainActivity.kt
│   │   │   └── MainViewModel.kt
│   │   ├── camera/
│   │   │   ├── CameraFragment.kt
│   │   │   └── CameraViewModel.kt
│   │   └── settings/
│   │       ├── SettingsFragment.kt
│   │       └── SettingsViewModel.kt
│   └── components/          # UI components
│       ├── SensorVisualization.kt
│       └── ConnectionStatus.kt
└── utils/
    ├── SensorManager.kt
    ├── EdgeAIProcessor.kt
    ├── AudioProcessor.kt
    └── TimestampSync.kt
```

---

## 📡 Módulo 1: Gerenciamento de Sensores

### 1.1 SensorDataCollector

Classe centralizada para coletar dados de todos os sensores com timestamp sincronizado.

```kotlin
class SensorDataCollector(
    private val context: Context,
    private val coroutineScope: CoroutineScope
) : SensorEventListener {

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as android.hardware.SensorManager

    // Sensores
    private val magnetometer = sensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD)
    private val accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)

    // Buffer circular para dados
    private val dataBuffer = CircularBuffer<SensorPacket>(capacity = 500) // 5 segundos @ 100Hz

    // Flow para emitir dados
    private val _sensorDataFlow = MutableSharedFlow<SensorPacket>(
        replay = 1,
        onBufferOverflow = BufferOverflow.DROP_OLDEST
    )
    val sensorDataFlow: SharedFlow<SensorPacket> = _sensorDataFlow.asSharedFlow()

    // Timestamp unificado
    private var startTimeNano: Long = 0L

    fun start() {
        startTimeNano = System.nanoTime()

        // Registrar sensores a 100Hz (10000 microsegundos)
        sensorManager.registerListener(this, magnetometer, 10_000)
        sensorManager.registerListener(this, accelerometer, 10_000)
        sensorManager.registerListener(this, gyroscope, 10_000)
    }

    fun stop() {
        sensorManager.unregisterListener(this)
    }

    override fun onSensorChanged(event: SensorEvent) {
        val timestamp = System.nanoTime()
        val relativeTime = timestamp - startTimeNano

        when (event.sensor.type) {
            Sensor.TYPE_MAGNETIC_FIELD -> {
                val magnitude = sqrt(event.values[0].pow(2) + event.values[1].pow(2) + event.values[2].pow(2))

                coroutineScope.launch {
                    _sensorDataFlow.emit(
                        SensorPacket(
                            timestamp = timestamp,
                            magneticData = MagneticData(
                                x = event.values[0],
                                y = event.values[1],
                                z = event.values[2],
                                magnitude = magnitude
                            )
                        )
                    )
                }
            }
            // ... outros sensores
        }
    }
}
```

### 1.2 Magnetômetro - Detecção de Anomalias

```kotlin
class MagneticAnomalyDetector(private val windowSize: Int = 100) {

    private val magnitudeHistory = ArrayDeque<Float>(windowSize)

    data class AnomalyResult(
        val isAnomaly: Boolean,
        val magnitude: Float,
        val mean: Float,
        val stdDev: Float,
        val threshold: Float
    )

    fun analyze(magnitude: Float): AnomalyResult {
        magnitudeHistory.addLast(magnitude)

        if (magnitudeHistory.size > windowSize) {
            magnitudeHistory.removeFirst()
        }

        if (magnitudeHistory.size < 10) {
            return AnomalyResult(false, magnitude, 0f, 0f, 0f)
        }

        val mean = magnitudeHistory.average().toFloat()
        val variance = magnitudeHistory.map { (it - mean).pow(2) }.average()
        val stdDev = sqrt(variance).toFloat()

        val threshold = mean + (3 * stdDev)
        val isAnomaly = magnitude > threshold

        return AnomalyResult(isAnomaly, magnitude, mean, stdDev, threshold)
    }
}
```

### 1.3 Fusão de Sensores (Acelerômetro + Giroscópio)

Implementação do **Filtro de Madgwick** para orientação robusta.

```kotlin
class SensorFusion {

    private var q0 = 1.0f
    private var q1 = 0.0f
    private var q2 = 0.0f
    private var q3 = 0.0f

    private val beta = 0.1f  // Ganho do filtro

    data class Orientation(
        val roll: Float,   // Rotação em X
        val pitch: Float,  // Rotação em Y
        val yaw: Float     // Rotação em Z (direção da câmera)
    )

    fun update(
        accel: FloatArray,  // [x, y, z]
        gyro: FloatArray,   // [x, y, z] em rad/s
        dt: Float           // Delta time em segundos
    ): Orientation {

        // Normalizar acelerômetro
        val norm = sqrt(accel[0].pow(2) + accel[1].pow(2) + accel[2].pow(2))
        val ax = accel[0] / norm
        val ay = accel[1] / norm
        val az = accel[2] / norm

        // Gradiente descendente para correção
        val s0 = -2 * (q1 * (2 * q0 * q2 - 2 * q1 * q3 - ax) + q2 * (2 * q0 * q1 + 2 * q2 * q3 - ay) + q3 * (1 - 2 * q1 * q1 - 2 * q2 * q2 - az))
        val s1 = 2 * (q0 * (2 * q0 * q2 - 2 * q1 * q3 - ax) + q3 * (2 * q0 * q1 + 2 * q2 * q3 - ay) - 2 * q1 * (1 - 2 * q1 * q1 - 2 * q2 * q2 - az))
        val s2 = 2 * (q3 * (2 * q0 * q2 - 2 * q1 * q3 - ax) + q0 * (2 * q0 * q1 + 2 * q2 * q3 - ay) - 2 * q2 * (1 - 2 * q1 * q1 - 2 * q2 * q2 - az))
        val s3 = -2 * q2 * (2 * q0 * q2 - 2 * q1 * q3 - ax) + 2 * q1 * (2 * q0 * q1 + 2 * q2 * q3 - ay)

        // Integrar taxa de mudança do quaternion
        val qDot0 = 0.5f * (-q1 * gyro[0] - q2 * gyro[1] - q3 * gyro[2]) - beta * s0
        val qDot1 = 0.5f * (q0 * gyro[0] + q2 * gyro[2] - q3 * gyro[1]) - beta * s1
        val qDot2 = 0.5f * (q0 * gyro[1] - q1 * gyro[2] + q3 * gyro[0]) - beta * s2
        val qDot3 = 0.5f * (q0 * gyro[2] + q1 * gyro[1] - q2 * gyro[0]) - beta * s3

        // Atualizar quaternion
        q0 += qDot0 * dt
        q1 += qDot1 * dt
        q2 += qDot2 * dt
        q3 += qDot3 * dt

        // Normalizar quaternion
        val qNorm = sqrt(q0.pow(2) + q1.pow(2) + q2.pow(2) + q3.pow(2))
        q0 /= qNorm
        q1 /= qNorm
        q2 /= qNorm
        q3 /= qNorm

        // Converter para ângulos de Euler
        return Orientation(
            roll = atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1.pow(2) + q2.pow(2))),
            pitch = asin(2 * (q0 * q2 - q3 * q1)),
            yaw = atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2.pow(2) + q3.pow(2)))
        )
    }
}
```

---

## 🎙️ Módulo 2: Processamento de Áudio

### 2.1 AudioRecorder com Espectrograma

```kotlin
class AudioProcessor(private val sampleRate: Int = 44100) {

    private val audioRecord: AudioRecord
    private val bufferSize = AudioRecord.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT
    )

    private val audioBuffer = ShortArray(bufferSize)
    private var isRecording = false

    // FFT para espectrograma
    private val noise = Noise.real(bufferSize)

    data class AudioAnalysis(
        val waveform: FloatArray,
        val spectrum: FloatArray,
        val peakFrequency: Float,
        val peakAmplitude: Float
    )

    init {
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize
        )
    }

    fun startRecording(callback: (AudioAnalysis) -> Unit) {
        isRecording = true
        audioRecord.startRecording()

        Thread {
            while (isRecording) {
                val readSize = audioRecord.read(audioBuffer, 0, bufferSize)

                if (readSize > 0) {
                    val analysis = analyzeAudio(audioBuffer)
                    callback(analysis)
                }
            }
        }.start()
    }

    private fun analyzeAudio(buffer: ShortArray): AudioAnalysis {
        // Converter para float normalizado
        val waveform = FloatArray(buffer.size) { buffer[it] / 32768.0f }

        // Aplicar janela de Hanning
        val windowed = FloatArray(waveform.size) { i ->
            val window = 0.5f * (1 - cos(2 * PI.toFloat() * i / waveform.size))
            waveform[i] * window
        }

        // FFT
        val fftResult = noise.fft(windowed, FloatArray(windowed.size))

        // Calcular magnitude do espectro
        val spectrum = FloatArray(fftResult.size / 2) { i ->
            val real = fftResult[i * 2]
            val imag = fftResult[i * 2 + 1]
            sqrt(real * real + imag * imag)
        }

        // Encontrar pico
        val peakIndex = spectrum.indices.maxByOrNull { spectrum[it] } ?: 0
        val peakFrequency = peakIndex * sampleRate.toFloat() / spectrum.size
        val peakAmplitude = spectrum[peakIndex]

        return AudioAnalysis(waveform, spectrum, peakFrequency, peakAmplitude)
    }

    fun stopRecording() {
        isRecording = false
        audioRecord.stop()
    }
}
```

---

## 📹 Módulo 3: Câmera e Vídeo

### 3.1 CameraX com Timestamp de Hardware

```kotlin
class CameraController(
    private val lifecycleOwner: LifecycleOwner,
    private val context: Context
) {

    private var videoCapture: VideoCapture<Recorder>? = null
    private var recording: Recording? = null

    data class VideoFrame(
        val timestamp: Long,
        val imageProxy: ImageProxy,
        val orientation: SensorFusion.Orientation
    )

    fun startCamera(
        previewView: PreviewView,
        onFrameAvailable: (VideoFrame) -> Unit
    ) {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(context)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .setTargetResolution(Size(1920, 1080))  // 2K
                .build()
                .also {
                    it.setSurfaceProvider(previewView.surfaceProvider)
                }

            // ImageAnalysis para frames
            val imageAnalyzer = ImageAnalysis.Builder()
                .setTargetResolution(Size(1920, 1080))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(Executors.newSingleThreadExecutor()) { imageProxy ->
                        // Timestamp de hardware
                        val timestamp = imageProxy.imageInfo.timestamp

                        onFrameAvailable(
                            VideoFrame(
                                timestamp = timestamp,
                                imageProxy = imageProxy,
                                orientation = getCurrentOrientation()
                            )
                        )

                        imageProxy.close()
                    }
                }

            // Recorder para salvar vídeo
            val recorder = Recorder.Builder()
                .setQualitySelector(QualitySelector.from(Quality.FHD))
                .build()

            videoCapture = VideoCapture.withOutput(recorder)

            // Bind to lifecycle
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    lifecycleOwner,
                    cameraSelector,
                    preview,
                    imageAnalyzer,
                    videoCapture
                )
            } catch (exc: Exception) {
                Timber.e(exc, "Camera binding failed")
            }

        }, ContextCompat.getMainExecutor(context))
    }

    fun startRecording(outputFile: File) {
        val videoCapture = videoCapture ?: return

        val outputOptions = FileOutputOptions.Builder(outputFile).build()

        recording = videoCapture.output
            .prepareRecording(context, outputOptions)
            .start(ContextCompat.getMainExecutor(context)) { recordEvent ->
                when (recordEvent) {
                    is VideoRecordEvent.Finalize -> {
                        if (!recordEvent.hasError()) {
                            Timber.d("Video saved: ${outputFile.absolutePath}")
                        }
                    }
                }
            }
    }

    fun stopRecording() {
        recording?.stop()
        recording = null
    }
}
```

---

## 🧠 Módulo 4: Edge AI (NPU)

### 4.1 Pose Estimation com TensorFlow Lite

```kotlin
class PoseEstimator(private val context: Context) {

    private lateinit var interpreter: Interpreter
    private val inputSize = 192  // MoveNet Lightning

    data class KeyPoint(
        val x: Float,
        val y: Float,
        val confidence: Float
    )

    data class PoseResult(
        val keypoints: List<KeyPoint>,
        val humanoidDetected: Boolean,
        val confidence: Float
    )

    init {
        loadModel()
    }

    private fun loadModel() {
        val modelFile = loadModelFile("movenet_lightning.tflite")

        val options = Interpreter.Options().apply {
            // Usar NPU se disponível (MediaTek APU)
            setUseNNAPI(true)
            setNumThreads(4)
        }

        interpreter = Interpreter(modelFile, options)
    }

    private fun loadModelFile(filename: String): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun estimatePose(bitmap: Bitmap): PoseResult {
        // Pré-processar imagem
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
        val inputBuffer = convertBitmapToByteBuffer(resizedBitmap)

        // Output: [1, 1, 17, 3] - 17 keypoints com (y, x, confidence)
        val outputArray = Array(1) { Array(1) { Array(17) { FloatArray(3) } } }

        // Inferência
        interpreter.run(inputBuffer, outputArray)

        // Processar resultados
        val keypoints = outputArray[0][0].map { kp ->
            KeyPoint(
                x = kp[1],
                y = kp[0],
                confidence = kp[2]
            )
        }

        // Detectar humanoide: pelo menos 5 keypoints com confidence > 0.3
        val validKeypoints = keypoints.count { it.confidence > 0.3f }
        val humanoidDetected = validKeypoints >= 5
        val avgConfidence = keypoints.map { it.confidence }.average().toFloat()

        return PoseResult(keypoints, humanoidDetected, avgConfidence)
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(inputSize * inputSize)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        var pixel = 0
        for (i in 0 until inputSize) {
            for (j in 0 until inputSize) {
                val value = intValues[pixel++]
                // Normalizar para [0, 1]
                byteBuffer.putFloat(((value shr 16) and 0xFF) / 255.0f)  // R
                byteBuffer.putFloat(((value shr 8) and 0xFF) / 255.0f)   // G
                byteBuffer.putFloat((value and 0xFF) / 255.0f)            // B
            }
        }

        return byteBuffer
    }

    fun close() {
        interpreter.close()
    }
}
```

---

## 🌐 Módulo 5: Transmissão de Dados (Ktor WebSocket)

### 5.1 WebSocket Client

```kotlin
@Serializable
data class SensorPacket(
    val timestamp: Long,
    val deviceId: String,
    val magnetometer: MagneticData? = null,
    val accelerometer: Vector3? = null,
    val gyroscope: Vector3? = null,
    val orientation: OrientationData? = null,
    val audioPeak: Float? = null,
    val humanoidDetected: Boolean = false,
    val bluetoothDevicesCount: Int = 0
)

@Serializable
data class MagneticData(
    val x: Float,
    val y: Float,
    val z: Float,
    val magnitude: Float
)

@Serializable
data class Vector3(val x: Float, val y: Float, val z: Float)

@Serializable
data class OrientationData(val roll: Float, val pitch: Float, val yaw: Float)

class WebSocketClient(private val serverUrl: String) {

    private val client = HttpClient {
        install(WebSockets) {
            pingInterval = 20_000  // Ping a cada 20s
        }
        install(Logging) {
            logger = Logger.DEFAULT
            level = LogLevel.INFO
        }
    }

    private var session: DefaultClientWebSocketSession? = null
    private val json = Json { ignoreUnknownKeys = true }

    suspend fun connect(
        onConnected: () -> Unit,
        onMessage: (String) -> Unit,
        onDisconnected: () -> Unit
    ) {
        try {
            client.webSocket(serverUrl) {
                session = this
                onConnected()

                try {
                    for (frame in incoming) {
                        when (frame) {
                            is Frame.Text -> {
                                val message = frame.readText()
                                onMessage(message)
                            }
                            else -> {}
                        }
                    }
                } catch (e: Exception) {
                    Timber.e(e, "WebSocket error")
                } finally {
                    onDisconnected()
                }
            }
        } catch (e: Exception) {
            Timber.e(e, "Connection failed")
            onDisconnected()
        }
    }

    suspend fun sendPacket(packet: SensorPacket) {
        try {
            val jsonString = json.encodeToString(SensorPacket.serializer(), packet)
            session?.send(Frame.Text(jsonString))
        } catch (e: Exception) {
            Timber.e(e, "Failed to send packet")
        }
    }

    fun disconnect() {
        session?.cancel()
        client.close()
    }
}
```

---

## 🔋 Otimização de Bateria

### Power Management

```kotlin
class PowerOptimizer(private val context: Context) {

    fun enableLowPowerMode() {
        // Reduzir taxa de sensores para 50Hz
        // Desabilitar pose estimation temporariamente
        // Reduzir resolução de vídeo para 720p
    }

    fun monitorBatteryLevel(): Flow<Int> = callbackFlow {
        val batteryReceiver = object : BroadcastReceiver() {
            override fun onReceive(context: Context, intent: Intent) {
                val level = intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)
                val scale = intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1)
                val batteryPct = (level / scale.toFloat() * 100).toInt()

                trySend(batteryPct)

                // Auto low-power mode abaixo de 20%
                if (batteryPct < 20) {
                    enableLowPowerMode()
                }
            }
        }

        context.registerReceiver(
            batteryReceiver,
            IntentFilter(Intent.ACTION_BATTERY_CHANGED)
        )

        awaitClose {
            context.unregisterReceiver(batteryReceiver)
        }
    }
}
```

---

## 📊 Performance Targets

| Métrica | Target | Crítico |
|---------|--------|---------|
| **Taxa de Coleta** | 100Hz | Todos os sensores |
| **Taxa de Transmissão** | 10Hz | WebSocket |
| **Latência de Rede** | < 50ms | 95th percentile |
| **Uso de CPU** | < 30% | Média |
| **Uso de Bateria** | < 15%/hora | Com todos sensores ativos |
| **Edge AI Latency** | < 100ms | Por frame |

---

## 🔐 Permissões Necessárias (AndroidManifest.xml)

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.RECORD_AUDIO" />
<uses-permission android:name="android.permission.INTERNET" />
<uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
<uses-permission android:name="android.permission.BLUETOOTH" />
<uses-permission android:name="android.permission.BLUETOOTH_ADMIN" />
<uses-permission android:name="android.permission.BLUETOOTH_SCAN" />
<uses-permission android:name="android.permission.NFC" />
<uses-permission android:name="android.permission.WAKE_LOCK" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
```

---

**Última Atualização**: 2025-01-17
**Versão**: 1.0.0
**Target Device**: OPPO Reno 11 F5 (MediaTek Dimensity 7050)
