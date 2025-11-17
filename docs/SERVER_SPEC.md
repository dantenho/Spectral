# 🖥️ Especificação Técnica - Servidor Backend

## 🎯 Visão Geral

Servidor backend desenvolvido em Python para receber dados de múltiplos clientes Android em tempo real, detectar anomalias correlacionadas, armazenar eventos estruturados e executar pipeline de treinamento de IA usando GPU NVIDIA RTX 4090.

---

## 🔧 Stack Tecnológico

### Core
- **Linguagem**: Python 3.11+
- **Framework Web**: FastAPI 0.109+
- **ASGI Server**: Uvicorn com Uvloop
- **Concorrência**: asyncio + multiprocessing

### Bibliotecas Principais

```python
# requirements.txt

# Web Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
websockets==12.0
pydantic==2.5.3
python-multipart==0.0.6

# Data Processing
numpy==1.26.3
scipy==1.11.4
pandas==2.1.4

# Audio Processing
librosa==0.10.1
soundfile==0.12.1
pydub==0.25.1

# Video Processing
opencv-python==4.9.0.80
av==11.0.0

# Database
influxdb-client==1.39.0
psycopg2-binary==2.9.9
sqlalchemy==2.0.25
alembic==1.13.1

# Deep Learning
torch==2.1.2+cu121
torchvision==0.16.2+cu121
torchaudio==2.1.2+cu121
pytorch-lightning==2.1.3
timm==0.9.12  # Pre-trained models
tensorboard==2.15.1

# ML Utilities
scikit-learn==1.4.0
wandb==0.16.2  # Weights & Biases

# Visualization
plotly==5.18.0
matplotlib==3.8.2

# Interface
gradio==4.13.0

# Utilities
python-dotenv==1.0.0
loguru==0.7.2
tqdm==4.66.1
```

---

## 🏗️ Arquitetura do Servidor

```
server/
├── main.py                      # Entry point
├── config/
│   ├── __init__.py
│   ├── settings.py              # Configurações
│   └── logging.py               # Setup de logs
├── api/
│   ├── __init__.py
│   ├── websocket.py             # WebSocket handlers
│   ├── rest.py                  # REST endpoints
│   └── schemas.py               # Pydantic models
├── core/
│   ├── __init__.py
│   ├── anomaly_detection.py    # Engine de detecção
│   ├── event_manager.py         # Gerenciador de eventos
│   ├── buffer_manager.py        # Buffers circulares
│   └── correlation_engine.py    # Correlação multi-sensorial
├── processing/
│   ├── __init__.py
│   ├── audio_processor.py       # FFT, EVP analysis
│   ├── video_processor.py       # Frame extraction
│   ├── magnetic_processor.py    # Análise magnética
│   └── sensor_fusion.py         # Fusão de dados
├── database/
│   ├── __init__.py
│   ├── influxdb_client.py       # Time series
│   ├── postgres_client.py       # Relacional
│   └── models.py                # SQLAlchemy models
├── ml/
│   ├── __init__.py
│   ├── dataset.py               # PyTorch Dataset
│   ├── models/
│   │   ├── __init__.py
│   │   ├── video_encoder.py     # EfficientNet
│   │   ├── audio_encoder.py     # 1D CNN
│   │   └── fusion_classifier.py # MLP/LSTM
│   ├── training.py              # Lightning Module
│   └── inference.py             # Modelo em produção
├── storage/
│   ├── __init__.py
│   ├── event_storage.py         # Sistema de arquivos
│   └── video_buffer.py          # Buffer de vídeo
└── interface/
    ├── __init__.py
    └── gradio_app.py            # Painel Gradio
```

---

## 🌐 Módulo 1: WebSocket Handler

### 1.1 FastAPI WebSocket Server

```python
# api/websocket.py

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
from loguru import logger
from datetime import datetime

class ConnectionManager:
    """Gerencia conexões WebSocket de múltiplos clientes"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_buffers: Dict[str, list] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_buffers[client_id] = []
        logger.info(f"Client {client_id} connected. Total clients: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            del self.client_buffers[client_id]
            logger.info(f"Client {client_id} disconnected. Total clients: {len(self.active_connections)}")

    async def send_message(self, client_id: str, message: dict):
        websocket = self.active_connections.get(client_id)
        if websocket:
            await websocket.send_json(message)

    async def broadcast(self, message: dict):
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")


manager = ConnectionManager()


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)

    try:
        while True:
            # Receber pacote JSON
            data = await websocket.receive_json()

            # Processar dados
            await process_sensor_data(client_id, data)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)
```

### 1.2 Pydantic Schemas

```python
# api/schemas.py

from pydantic import BaseModel, Field
from typing import Optional

class MagneticData(BaseModel):
    x: float
    y: float
    z: float
    magnitude: float

class Vector3(BaseModel):
    x: float
    y: float
    z: float

class OrientationData(BaseModel):
    roll: float
    pitch: float
    yaw: float

class SensorPacket(BaseModel):
    timestamp: int = Field(..., description="Nanoseconds since epoch")
    device_id: str
    magnetometer: Optional[MagneticData] = None
    accelerometer: Optional[Vector3] = None
    gyroscope: Optional[Vector3] = None
    orientation: Optional[OrientationData] = None
    audio_peak: Optional[float] = None
    humanoid_detected: bool = False
    bluetooth_devices_count: int = 0

    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": 1678886400000000000,
                "device_id": "OPPO_Reno_11_01",
                "magnetometer": {"x": 0.1, "y": -0.5, "z": 0.4, "magnitude": 0.648},
                "accelerometer": {"x": 0.01, "y": 0.98, "z": 0.02},
                "gyroscope": {"x": 0.001, "y": -0.002, "z": 0.005},
                "orientation": {"roll": 0.0, "pitch": 1.57, "yaw": 0.0},
                "audio_peak": 0.75,
                "humanoid_detected": False,
                "bluetooth_devices_count": 3
            }
        }
```

---

## 🔍 Módulo 2: Anomaly Detection Engine

### 2.1 Detecção de Anomalia Magnética

```python
# core/anomaly_detection.py

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional

@dataclass
class AnomalyResult:
    is_anomaly: bool
    magnitude: float
    mean: float
    std_dev: float
    threshold: float
    z_score: float

class MagneticAnomalyDetector:
    """Detecta anomalias magnéticas usando média móvel e desvio padrão"""

    def __init__(self, window_size: int = 100, sigma_multiplier: float = 3.0):
        self.window_size = window_size
        self.sigma_multiplier = sigma_multiplier
        self.magnitude_history = deque(maxlen=window_size)

    def analyze(self, magnitude: float) -> AnomalyResult:
        self.magnitude_history.append(magnitude)

        if len(self.magnitude_history) < 10:
            return AnomalyResult(
                is_anomaly=False,
                magnitude=magnitude,
                mean=0.0,
                std_dev=0.0,
                threshold=0.0,
                z_score=0.0
            )

        # Estatísticas
        magnitudes = np.array(self.magnitude_history)
        mean = np.mean(magnitudes)
        std_dev = np.std(magnitudes)

        # Threshold = média + (sigma_multiplier * desvio padrão)
        threshold = mean + (self.sigma_multiplier * std_dev)

        # Z-score
        z_score = (magnitude - mean) / std_dev if std_dev > 0 else 0

        is_anomaly = magnitude > threshold

        return AnomalyResult(
            is_anomaly=is_anomaly,
            magnitude=magnitude,
            mean=mean,
            std_dev=std_dev,
            threshold=threshold,
            z_score=z_score
        )
```

### 2.2 Análise de Áudio com FFT

```python
# processing/audio_processor.py

import numpy as np
import librosa
from scipy import signal
from typing import Tuple, List

class AudioAnomalyDetector:
    """Detecta anomalias em áudio usando FFT e análise espectral"""

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def analyze_spectrum(
        self,
        audio_data: np.ndarray
    ) -> Tuple[bool, dict]:
        """
        Analisa espectro de frequência para detectar anomalias

        Returns:
            (is_anomaly, analysis_dict)
        """
        # FFT
        frequencies, power_spectrum = signal.welch(
            audio_data,
            fs=self.sample_rate,
            nperseg=1024
        )

        # Converter para dB
        power_db = 10 * np.log10(power_spectrum + 1e-10)

        # Detectar picos anômalos
        anomalies = {
            'infrasound': False,  # < 20 Hz
            'ultrasound': False,  # > 18 kHz
            'formants': False     # Formantes de voz
        }

        # Infrasound (< 20 Hz)
        infrasound_mask = frequencies < 20
        if np.any(infrasound_mask):
            infrasound_power = np.max(power_db[infrasound_mask])
            if infrasound_power > -40:  # Threshold ajustável
                anomalies['infrasound'] = True

        # Ultrasound (> 18 kHz)
        ultrasound_mask = frequencies > 18000
        if np.any(ultrasound_mask):
            ultrasound_power = np.max(power_db[ultrasound_mask])
            if ultrasound_power > -40:
                anomalies['ultrasound'] = True

        # Detecção de formantes (voz)
        formants = self.detect_formants(audio_data)
        if formants is not None:
            anomalies['formants'] = True

        is_anomaly = any(anomalies.values())

        analysis = {
            'anomalies': anomalies,
            'peak_frequency': frequencies[np.argmax(power_spectrum)],
            'peak_power_db': np.max(power_db),
            'formants': formants
        }

        return is_anomaly, analysis

    def detect_formants(self, audio_data: np.ndarray) -> Optional[List[float]]:
        """Detecta formantes usando LPC (Linear Predictive Coding)"""
        try:
            # LPC para modelar trato vocal
            lpc_order = 12
            lpc_coeffs = librosa.lpc(audio_data, order=lpc_order)

            # Encontrar raízes do polinômio LPC
            roots = np.roots(lpc_coeffs)

            # Converter para frequências
            angles = np.angle(roots)
            frequencies = angles * (self.sample_rate / (2 * np.pi))

            # Filtrar apenas frequências positivas
            formants = sorted([f for f in frequencies if 0 < f < self.sample_rate / 2])

            # Formantes de voz humana: F1 (250-1000 Hz), F2 (700-3800 Hz)
            valid_formants = [f for f in formants if 250 < f < 3800]

            if len(valid_formants) >= 2:
                return valid_formants[:3]  # F1, F2, F3

        except Exception as e:
            logger.error(f"Formant detection error: {e}")

        return None
```

### 2.3 Correlation Engine

```python
# core/correlation_engine.py

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

@dataclass
class CorrelationEvent:
    timestamp: datetime
    magnetic_anomaly: bool
    audio_anomaly: bool
    humanoid_detected: bool
    correlation_score: float

class CorrelationEngine:
    """Detecta correlação entre múltiplos sensores em janela de tempo"""

    def __init__(self, time_window_ms: int = 1000):
        self.time_window = timedelta(milliseconds=time_window_ms)
        self.event_buffer: List[dict] = []

    def add_event(self, event_data: dict) -> Optional[CorrelationEvent]:
        """
        Adiciona evento e verifica correlação

        Returns:
            CorrelationEvent se anomalia correlacionada detectada
        """
        current_time = datetime.fromtimestamp(event_data['timestamp'] / 1e9)

        # Limpar eventos antigos
        self.event_buffer = [
            e for e in self.event_buffer
            if current_time - datetime.fromtimestamp(e['timestamp'] / 1e9) < self.time_window
        ]

        # Adicionar novo evento
        self.event_buffer.append(event_data)

        # Verificar correlação
        magnetic_anomaly = event_data.get('magnetic_anomaly', False)
        audio_anomaly = event_data.get('audio_anomaly', False)
        humanoid_detected = event_data.get('humanoid_detected', False)

        # Scoring
        score = 0.0
        if magnetic_anomaly:
            score += 0.4
        if audio_anomaly:
            score += 0.4
        if humanoid_detected:
            score += 0.2

        # Trigger: pelo menos 2 anomalias simultâneas
        is_correlated = score >= 0.6

        if is_correlated:
            return CorrelationEvent(
                timestamp=current_time,
                magnetic_anomaly=magnetic_anomaly,
                audio_anomaly=audio_anomaly,
                humanoid_detected=humanoid_detected,
                correlation_score=score
            )

        return None
```

---

## 💾 Módulo 3: Event Storage System

### 3.1 Event Manager

```python
# core/event_manager.py

import json
import shutil
from pathlib import Path
from datetime import datetime
import cv2
import soundfile as sf
import pandas as pd
from typing import List, Dict

class EventManager:
    """Gerencia armazenamento de eventos anômalos"""

    def __init__(self, storage_dir: Path):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_event(
        self,
        event_id: str,
        video_frames: List[np.ndarray],
        audio_data: np.ndarray,
        sensor_data: pd.DataFrame,
        metadata: dict
    ) -> Path:
        """
        Salva evento completo com todos os dados

        Estrutura:
        events/
        └── event_20250117_143052/
            ├── video.mp4
            ├── audio.wav
            ├── sensors.csv
            └── metadata.json
        """
        # Criar diretório do evento
        event_dir = self.storage_dir / event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        # Salvar vídeo
        video_path = event_dir / "video.mp4"
        self._save_video(video_frames, video_path)

        # Salvar áudio
        audio_path = event_dir / "audio.wav"
        self._save_audio(audio_data, audio_path)

        # Salvar sensores
        sensor_path = event_dir / "sensors.csv"
        sensor_data.to_csv(sensor_path, index=False)

        # Salvar metadata
        metadata_path = event_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Event {event_id} saved to {event_dir}")
        return event_dir

    def _save_video(self, frames: List[np.ndarray], output_path: Path):
        """Salva frames como vídeo H.265"""
        if not frames:
            return

        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30

        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )

        for frame in frames:
            out.write(frame)

        out.release()

    def _save_audio(self, audio_data: np.ndarray, output_path: Path):
        """Salva áudio RAW como WAV"""
        sf.write(
            str(output_path),
            audio_data,
            samplerate=44100,
            subtype='PCM_16'
        )

    def load_event(self, event_id: str) -> Dict:
        """Carrega evento salvo"""
        event_dir = self.storage_dir / event_id

        if not event_dir.exists():
            raise ValueError(f"Event {event_id} not found")

        # Carregar metadata
        with open(event_dir / "metadata.json") as f:
            metadata = json.load(f)

        # Carregar sensor data
        sensor_data = pd.read_csv(event_dir / "sensors.csv")

        return {
            'metadata': metadata,
            'sensor_data': sensor_data,
            'video_path': event_dir / "video.mp4",
            'audio_path': event_dir / "audio.wav"
        }
```

---

## 🗄️ Módulo 4: Database Layer

### 4.1 InfluxDB Client (Time Series)

```python
# database/influxdb_client.py

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from datetime import datetime

class InfluxDBHandler:
    """Cliente para InfluxDB - armazena time series de sensores"""

    def __init__(self, url: str, token: str, org: str, bucket: str):
        self.client = InfluxDBClient(url=url, token=token, org=org)
        self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
        self.query_api = self.client.query_api()
        self.bucket = bucket
        self.org = org

    def write_sensor_data(self, device_id: str, sensor_data: dict):
        """Escreve dados de sensores como time series"""
        timestamp = datetime.fromtimestamp(sensor_data['timestamp'] / 1e9)

        # Magnetic data
        if sensor_data.get('magnetometer'):
            mag = sensor_data['magnetometer']
            point = (
                Point("magnetic")
                .tag("device_id", device_id)
                .field("x", mag['x'])
                .field("y", mag['y'])
                .field("z", mag['z'])
                .field("magnitude", mag['magnitude'])
                .time(timestamp, WritePrecision.NS)
            )
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)

        # Accelerometer
        if sensor_data.get('accelerometer'):
            acc = sensor_data['accelerometer']
            point = (
                Point("accelerometer")
                .tag("device_id", device_id)
                .field("x", acc['x'])
                .field("y", acc['y'])
                .field("z", acc['z'])
                .time(timestamp, WritePrecision.NS)
            )
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)

        # Gyroscope
        if sensor_data.get('gyroscope'):
            gyro = sensor_data['gyroscope']
            point = (
                Point("gyroscope")
                .tag("device_id", device_id)
                .field("x", gyro['x'])
                .field("y", gyro['y'])
                .field("z", gyro['z'])
                .time(timestamp, WritePrecision.NS)
            )
            self.write_api.write(bucket=self.bucket, org=self.org, record=point)

    def query_time_range(
        self,
        device_id: str,
        start: datetime,
        end: datetime,
        measurement: str = "magnetic"
    ):
        """Query dados em intervalo de tempo"""
        query = f'''
        from(bucket: "{self.bucket}")
            |> range(start: {start.isoformat()}Z, stop: {end.isoformat()}Z)
            |> filter(fn: (r) => r["_measurement"] == "{measurement}")
            |> filter(fn: (r) => r["device_id"] == "{device_id}")
        '''
        result = self.query_api.query(org=self.org, query=query)
        return result
```

### 4.2 PostgreSQL Client (Events)

```python
# database/models.py

from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(100), unique=True, nullable=False, index=True)
    device_id = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Anomaly flags
    magnetic_anomaly = Column(Boolean, default=False)
    audio_anomaly = Column(Boolean, default=False)
    humanoid_detected = Column(Boolean, default=False)

    # Scores
    correlation_score = Column(Float, default=0.0)
    magnetic_magnitude = Column(Float)
    audio_peak_db = Column(Float)

    # Paths
    storage_path = Column(String(255))

    # Metadata
    metadata = Column(JSON)

    # Classification (from AI)
    classification = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Event {self.event_id} - {self.classification}>"
```

---

## 📊 Módulo 5: Main Application

### 5.1 FastAPI Main

```python
# main.py

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from config.settings import settings
from api.websocket import manager, process_sensor_data
from api.rest import router as rest_router
from database.influxdb_client import InfluxDBHandler
from database.postgres_client import get_db
from core.anomaly_detection import MagneticAnomalyDetector
from core.correlation_engine import CorrelationEngine
from core.event_manager import EventManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Spectral Server...")

    # Inicializar componentes
    app.state.influxdb = InfluxDBHandler(
        url=settings.INFLUXDB_URL,
        token=settings.INFLUXDB_TOKEN,
        org=settings.INFLUXDB_ORG,
        bucket=settings.INFLUXDB_BUCKET
    )

    app.state.event_manager = EventManager(storage_dir=settings.EVENT_STORAGE_DIR)

    logger.info("Spectral Server ready!")

    yield

    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(
    title="Spectral Backend",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(rest_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Spectral Backend API", "version": "1.0.0"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
```

---

## ⚙️ Configurações

```python
# config/settings.py

from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Server
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000

    # InfluxDB
    INFLUXDB_URL: str = "http://localhost:8086"
    INFLUXDB_TOKEN: str = ""
    INFLUXDB_ORG: str = "spectral"
    INFLUXDB_BUCKET: str = "sensors"

    # PostgreSQL
    POSTGRES_URL: str = "postgresql://user:pass@localhost:5432/spectral"

    # Storage
    EVENT_STORAGE_DIR: Path = Path("data/events")
    MODEL_STORAGE_DIR: Path = Path("models")

    # Anomaly Detection
    MAGNETIC_WINDOW_SIZE: int = 100
    MAGNETIC_SIGMA_MULTIPLIER: float = 3.0
    CORRELATION_TIME_WINDOW_MS: int = 1000

    # ML
    BATCH_SIZE: int = 16
    NUM_EPOCHS: int = 50
    LEARNING_RATE: float = 0.001

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
```

---

**Próximo**: docs/AI_ML_SPEC.md com especificação completa do pipeline de ML
**Última Atualização**: 2025-01-17
**Versão**: 1.0.0
**Hardware Target**: NVIDIA RTX 4090
