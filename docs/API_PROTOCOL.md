# 🔌 API e Protocolos de Comunicação

## 🎯 Visão Geral

Este documento especifica todos os protocolos de comunicação entre o Cliente Android, Servidor Backend e Interface Gradio no sistema Spectral.

---

## 🌐 Protocolo WebSocket

### Conexão

```
Cliente                                Servidor
   │                                      │
   │──── WebSocket Handshake ────────────▶│
   │     GET /ws/{client_id}              │
   │     Upgrade: websocket                │
   │                                      │
   │◀────── 101 Switching Protocols ──────│
   │                                      │
   │◀────── Connection Accepted ──────────│
   │     {"type": "connected"}            │
   │                                      │
```

### Endpoint

```
ws://[SERVER_IP]:8000/ws/{client_id}
```

**Parâmetros**:
- `client_id`: Identificador único do dispositivo (ex: "OPPO_Reno_11_01")

**Headers**:
```http
Connection: Upgrade
Upgrade: websocket
Sec-WebSocket-Version: 13
Sec-WebSocket-Key: [Base64 Random]
```

---

## 📦 Formato de Dados

### Pacote de Sensores (Cliente → Servidor)

Enviado a cada **100ms (10Hz)**

```json
{
  "timestamp": 1705501234567890000,
  "device_id": "OPPO_Reno_11_01",
  "magnetometer": {
    "x": 0.123,
    "y": -0.456,
    "z": 0.789,
    "magnitude": 0.936
  },
  "accelerometer": {
    "x": 0.012,
    "y": 9.801,
    "z": 0.034
  },
  "gyroscope": {
    "x": 0.0012,
    "y": -0.0034,
    "z": 0.0056
  },
  "orientation": {
    "roll": 0.052,
    "pitch": 1.571,
    "yaw": 0.123
  },
  "audio_peak": 0.753,
  "humanoid_detected": false,
  "bluetooth_devices_count": 3,
  "battery_level": 85
}
```

#### Campos Detalhados

| Campo | Tipo | Unidade | Descrição |
|-------|------|---------|-----------|
| `timestamp` | `int64` | nanosegundos | Timestamp Unix em nanosegundos |
| `device_id` | `string` | - | ID único do dispositivo |
| `magnetometer.x` | `float32` | µT | Campo magnético eixo X |
| `magnetometer.y` | `float32` | µT | Campo magnético eixo Y |
| `magnetometer.z` | `float32` | µT | Campo magnético eixo Z |
| `magnetometer.magnitude` | `float32` | µT | Magnitude do vetor |
| `accelerometer.{x,y,z}` | `float32` | m/s² | Aceleração linear |
| `gyroscope.{x,y,z}` | `float32` | rad/s | Velocidade angular |
| `orientation.roll` | `float32` | rad | Rotação em X |
| `orientation.pitch` | `float32` | rad | Rotação em Y |
| `orientation.yaw` | `float32` | rad | Rotação em Z (direção câmera) |
| `audio_peak` | `float32` | [0-1] | Amplitude normalizada |
| `humanoid_detected` | `bool` | - | Flag de pose estimation |
| `bluetooth_devices_count` | `int32` | - | Dispositivos BT detectados |
| `battery_level` | `int32` | % | Nível de bateria |

---

### Resposta do Servidor (Servidor → Cliente)

#### 1. Confirmação de Recebimento (ACK)

```json
{
  "type": "ack",
  "timestamp": 1705501234567890000,
  "status": "ok"
}
```

#### 2. Comando de Controle

```json
{
  "type": "command",
  "command": "adjust_sample_rate",
  "params": {
    "sample_rate_hz": 50
  }
}
```

**Comandos Disponíveis**:
- `adjust_sample_rate`: Alterar taxa de amostragem
- `enable_low_power`: Ativar modo economia
- `start_recording`: Iniciar gravação local
- `stop_recording`: Parar gravação

#### 3. Detecção de Anomalia

```json
{
  "type": "anomaly_detected",
  "event_id": "event_20250117_143052",
  "timestamp": 1705501234567890000,
  "correlation_score": 0.85,
  "details": {
    "magnetic_anomaly": true,
    "audio_anomaly": true,
    "humanoid_detected": false
  }
}
```

---

## 🎥 Transmissão de Vídeo

### Protocolo: WebRTC ou RTSP

#### Opção 1: WebRTC (Baixa Latência)

```javascript
// Cliente Android
RTCPeerConnection {
  offerToReceiveVideo: true,
  codec: "VP9",
  resolution: "1920x1080",
  framerate: 30
}
```

**URL**: `wss://[SERVER_IP]:8000/webrtc/{client_id}`

#### Opção 2: RTSP (Compatibilidade)

```
rtsp://[SERVER_IP]:8554/stream/{client_id}
```

**Formato**:
- Codec: H.265 (HEVC)
- Resolução: 1920x1080 (2K)
- FPS: 30
- Bitrate: 4 Mbps

---

## 🎵 Transmissão de Áudio

### Protocolo: WebSocket Binary

Áudio RAW enviado em chunks de **1 segundo**

```
Cliente                                Servidor
   │                                      │
   │──── Binary Frame (Audio Chunk) ─────▶│
   │     [44100 samples x 2 bytes]        │
   │     = 88.2 KB                        │
   │                                      │
```

**Formato**:
- Sample Rate: 44100 Hz
- Bit Depth: 16-bit PCM
- Channels: Mono
- Chunk Size: 1 segundo (88.2 KB)

**Encoding**:
```python
# Servidor recebe
audio_bytes = await websocket.receive_bytes()
audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
audio_float = audio_array.astype(np.float32) / 32768.0  # Normalizar
```

---

## 🔄 REST API Endpoints

### Base URL

```
http://[SERVER_IP]:8000/api/v1
```

### Endpoints

#### 1. Listar Eventos

```http
GET /api/v1/events
```

**Query Parameters**:
- `device_id` (optional): Filtrar por dispositivo
- `start_date` (optional): Data inicial (ISO 8601)
- `end_date` (optional): Data final
- `classification` (optional): Filtrar por classificação
- `limit` (default: 50): Número de resultados
- `offset` (default: 0): Paginação

**Response**:
```json
{
  "total": 127,
  "events": [
    {
      "event_id": "event_20250117_143052",
      "device_id": "OPPO_Reno_11_01",
      "timestamp": "2025-01-17T14:30:52.123456Z",
      "magnetic_anomaly": true,
      "audio_anomaly": true,
      "humanoid_detected": false,
      "correlation_score": 0.85,
      "classification": "Anomalia_Correlacionada",
      "confidence": 0.92,
      "storage_path": "data/events/event_20250117_143052"
    }
  ]
}
```

#### 2. Obter Evento Específico

```http
GET /api/v1/events/{event_id}
```

**Response**:
```json
{
  "event_id": "event_20250117_143052",
  "device_id": "OPPO_Reno_11_01",
  "timestamp": "2025-01-17T14:30:52.123456Z",
  "metadata": {
    "magnetic_magnitude": 95.3,
    "audio_peak_db": -12.5,
    "formants": [850, 1220, 2750]
  },
  "files": {
    "video": "/api/v1/events/event_20250117_143052/video",
    "audio": "/api/v1/events/event_20250117_143052/audio",
    "sensors": "/api/v1/events/event_20250117_143052/sensors"
  }
}
```

#### 3. Download de Arquivos do Evento

```http
GET /api/v1/events/{event_id}/video
GET /api/v1/events/{event_id}/audio
GET /api/v1/events/{event_id}/sensors
```

**Response**: Binary stream (video/mp4, audio/wav, text/csv)

#### 4. Classificar Evento (Manual)

```http
POST /api/v1/events/{event_id}/classify
```

**Request Body**:
```json
{
  "classification": "Interferência_Eletrônica",
  "confidence": 1.0,
  "notes": "Proximidade de roteador WiFi"
}
```

#### 5. Estatísticas em Tempo Real

```http
GET /api/v1/stats/realtime
```

**Response**:
```json
{
  "connected_clients": 3,
  "total_events_today": 47,
  "current_magnetic_avg": 48.5,
  "anomalies_last_hour": 12
}
```

#### 6. Query de Time Series (InfluxDB)

```http
POST /api/v1/timeseries/query
```

**Request Body**:
```json
{
  "device_id": "OPPO_Reno_11_01",
  "measurement": "magnetic",
  "start": "2025-01-17T14:00:00Z",
  "end": "2025-01-17T15:00:00Z",
  "fields": ["magnitude", "x", "y", "z"]
}
```

**Response**:
```json
{
  "data": [
    {
      "time": "2025-01-17T14:30:52.123Z",
      "magnitude": 48.5,
      "x": 12.3,
      "y": -15.6,
      "z": 45.2
    }
  ]
}
```

---

## 🧠 ML API Endpoints

### 1. Iniciar Treinamento

```http
POST /api/v1/ml/train
```

**Request Body**:
```json
{
  "dataset_path": "data/training/dataset_v1",
  "model_type": "fusion_classifier",
  "hyperparameters": {
    "batch_size": 16,
    "learning_rate": 0.001,
    "num_epochs": 50
  },
  "augmentation": true
}
```

**Response**:
```json
{
  "training_id": "train_20250117_150000",
  "status": "started",
  "estimated_time_minutes": 120
}
```

### 2. Status de Treinamento

```http
GET /api/v1/ml/train/{training_id}/status
```

**Response**:
```json
{
  "training_id": "train_20250117_150000",
  "status": "running",
  "current_epoch": 23,
  "total_epochs": 50,
  "metrics": {
    "train_loss": 0.234,
    "train_accuracy": 0.89,
    "val_loss": 0.312,
    "val_accuracy": 0.85
  },
  "elapsed_time_minutes": 45,
  "estimated_remaining_minutes": 52
}
```

### 3. Inferência em Evento

```http
POST /api/v1/ml/predict/{event_id}
```

**Request Body**:
```json
{
  "model_checkpoint": "models/best_model_v1.pth"
}
```

**Response**:
```json
{
  "event_id": "event_20250117_143052",
  "predictions": {
    "Ruído_Ambiente": 0.05,
    "Interferência_Eletrônica": 0.12,
    "Anomalia_Correlacionada": 0.78,
    "Forma_Humanoide_Potencial": 0.05
  },
  "top_prediction": "Anomalia_Correlacionada",
  "confidence": 0.78
}
```

---

## 📡 Protocolo de Sincronização

### Clock Sync (NTP-like)

Para sincronizar timestamps entre cliente e servidor:

```
Cliente                                Servidor
   │                                      │
   │──── Clock Sync Request (T1) ────────▶│
   │                                      │
   │◀──── Clock Sync Response (T2) ───────│
   │      {server_time: T2}               │
   │                                      │
   │  Offset = T2 - T1 - RTT/2            │
   │                                      │
```

**Request**:
```json
{
  "type": "clock_sync",
  "client_time": 1705501234567890000
}
```

**Response**:
```json
{
  "type": "clock_sync_response",
  "client_time": 1705501234567890000,
  "server_time": 1705501234568123000,
  "server_processing_time_ns": 123000
}
```

---

## 🔒 Autenticação e Segurança

### JWT Token

```http
POST /api/v1/auth/login
```

**Request**:
```json
{
  "device_id": "OPPO_Reno_11_01",
  "api_key": "your_api_key_here"
}
```

**Response**:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### WebSocket com Token

```
ws://[SERVER_IP]:8000/ws/{client_id}?token={jwt_token}
```

---

## 📊 Rate Limiting

| Endpoint | Limit |
|----------|-------|
| WebSocket Data | 10 packets/sec |
| REST API | 100 requests/min |
| Video Stream | 1 concurrent stream/client |
| ML Inference | 10 requests/min |

---

## ❌ Códigos de Erro

### WebSocket Errors

```json
{
  "type": "error",
  "code": "RATE_LIMIT_EXCEEDED",
  "message": "Too many packets. Limit: 10/sec",
  "timestamp": 1705501234567890000
}
```

**Códigos**:
- `RATE_LIMIT_EXCEEDED`: Taxa de envio excedida
- `INVALID_PACKET`: Formato JSON inválido
- `DEVICE_NOT_AUTHORIZED`: Device ID não autorizado
- `SERVER_OVERLOAD`: Servidor em sobrecarga

### HTTP Errors

```json
{
  "error": "EVENT_NOT_FOUND",
  "message": "Event event_xyz not found",
  "status_code": 404
}
```

---

## 🧪 Exemplos de Implementação

### Cliente Kotlin (Ktor)

```kotlin
suspend fun connectWebSocket() {
    client.webSocket(
        method = HttpMethod.Get,
        host = "192.168.1.100",
        port = 8000,
        path = "/ws/OPPO_Reno_11_01"
    ) {
        // Enviar pacote
        val packet = SensorPacket(
            timestamp = System.nanoTime(),
            deviceId = "OPPO_Reno_11_01",
            magnetometer = MagneticData(0.1f, -0.5f, 0.4f, 0.648f)
        )

        val json = Json.encodeToString(packet)
        send(Frame.Text(json))

        // Receber resposta
        for (frame in incoming) {
            if (frame is Frame.Text) {
                val response = frame.readText()
                println("Server: $response")
            }
        }
    }
}
```

### Servidor Python (FastAPI)

```python
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()

    try:
        while True:
            # Receber JSON
            data = await websocket.receive_json()

            # Validar com Pydantic
            packet = SensorPacket(**data)

            # Processar
            await process_packet(client_id, packet)

            # Enviar ACK
            await websocket.send_json({
                "type": "ack",
                "timestamp": packet.timestamp,
                "status": "ok"
            })

    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected")
```

---

## 📈 Métricas de Performance

| Métrica | Target | Monitoramento |
|---------|--------|---------------|
| Latência WebSocket | < 50ms | Prometheus |
| Throughput | 10 pacotes/seg/cliente | Grafana |
| Packet Loss | < 0.1% | InfluxDB |
| Video Latency | < 500ms | WebRTC stats |

---

**Última Atualização**: 2025-01-17
**Versão**: 1.0.0
**Protocolo Version**: v1
