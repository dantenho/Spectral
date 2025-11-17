"""
Servidor Principal do Spectral

Sistema de detecção de anomalias ambientais em tempo real
"""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# Configurações
from config.settings import settings, validate_settings

# Core
from core.kalman_filter import KalmanFilter1D, AdaptiveKalmanFilter
from processing.audio_variants import AudioEnsemble

# Importações condicionais (podem não existir ainda)
try:
    from database.influxdb_client import InfluxDBHandler
except ImportError:
    InfluxDBHandler = None
    logger.warning("InfluxDB client não disponível")

try:
    from database.postgres_client import PostgresHandler
except ImportError:
    PostgresHandler = None
    logger.warning("PostgreSQL client não disponível")


# ============================================================================
# GERENCIADOR DE CONEXÕES WEBSOCKET
# ============================================================================

class ConnectionManager:
    """Gerencia conexões WebSocket de múltiplos clientes"""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.kalman_filters: dict[str, KalmanFilter1D] = {}
        self.audio_processors: dict[str, AudioEnsemble] = {}

        # Buffers de dados para detecção de anomalia
        self.magnitude_buffers: dict[str, list[float]] = {}

        # Estatísticas
        self.stats = {
            'total_clients': 0,
            'total_packets_received': 0,
            'total_events_detected': 0,
            'clients_online': 0
        }

    async def connect(self, websocket: WebSocket, client_id: str):
        """Aceita e registra nova conexão"""
        await websocket.accept()

        self.active_connections[client_id] = websocket

        # Criar filtro de Kalman para este cliente
        self.kalman_filters[client_id] = AdaptiveKalmanFilter(
            process_variance=settings.KALMAN_PROCESS_VARIANCE,
            initial_measurement_variance=settings.KALMAN_MEASUREMENT_VARIANCE,
            adaptation_rate=0.1
        )

        # Criar processador de áudio
        self.audio_processors[client_id] = AudioEnsemble(
            sample_rate=settings.AUDIO_SAMPLE_RATE
        )

        # Inicializar buffer
        self.magnitude_buffers[client_id] = []

        # Atualizar stats
        self.stats['total_clients'] += 1
        self.stats['clients_online'] = len(self.active_connections)

        logger.info(f"✅ Cliente conectado: {client_id} (Total: {self.stats['clients_online']})")

        # Enviar mensagem de boas-vindas
        await self.send_message(client_id, {
            'type': 'connected',
            'message': f'Bem-vindo ao Spectral Server, {client_id}!',
            'server_version': '1.0.0',
            'timestamp': asyncio.get_event_loop().time()
        })

    def disconnect(self, client_id: str):
        """Remove conexão"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]

        if client_id in self.kalman_filters:
            del self.kalman_filters[client_id]

        if client_id in self.audio_processors:
            del self.audio_processors[client_id]

        if client_id in self.magnitude_buffers:
            del self.magnitude_buffers[client_id]

        self.stats['clients_online'] = len(self.active_connections)

        logger.info(f"❌ Cliente desconectado: {client_id} (Total: {self.stats['clients_online']})")

    async def send_message(self, client_id: str, message: dict):
        """Envia mensagem para cliente específico"""
        websocket = self.active_connections.get(client_id)
        if websocket:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Erro ao enviar mensagem para {client_id}: {e}")

    async def broadcast(self, message: dict):
        """Envia mensagem para todos os clientes"""
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Erro no broadcast para {client_id}: {e}")

    async def process_sensor_packet(self, client_id: str, packet: dict):
        """Processa pacote de sensores"""

        self.stats['total_packets_received'] += 1

        # Extrair dados
        magnetometer = packet.get('magnetometer')
        audio_peak = packet.get('audio_peak', 0.0)
        humanoid_detected = packet.get('humanoid_detected', False)

        anomaly_detected = False
        anomaly_details = {}

        # ====================================================================
        # PROCESSAR MAGNETÔMETRO COM KALMAN
        # ====================================================================
        if magnetometer:
            raw_magnitude = magnetometer.get('magnitude', 0.0)

            # Aplicar filtro de Kalman
            kalman = self.kalman_filters.get(client_id)
            if kalman:
                filtered_magnitude = kalman.process(raw_magnitude)

                # Adicionar ao buffer
                buffer = self.magnitude_buffers.get(client_id, [])
                buffer.append(filtered_magnitude)

                # Manter apenas últimas N amostras
                if len(buffer) > settings.MAGNETIC_WINDOW_SIZE:
                    buffer.pop(0)

                self.magnitude_buffers[client_id] = buffer

                # Detectar anomalia se tiver amostras suficientes
                if len(buffer) >= 10:
                    import numpy as np

                    mean = np.mean(buffer)
                    std = np.std(buffer)
                    threshold = mean + (settings.MAGNETIC_SIGMA_MULTIPLIER * std)

                    if filtered_magnitude > threshold:
                        anomaly_detected = True
                        anomaly_details['magnetic'] = {
                            'raw': raw_magnitude,
                            'filtered': filtered_magnitude,
                            'mean': float(mean),
                            'std': float(std),
                            'threshold': float(threshold),
                            'z_score': float((filtered_magnitude - mean) / std) if std > 0 else 0
                        }

        # ====================================================================
        # PROCESSAR ÁUDIO (se disponível)
        # ====================================================================
        # Nota: Para processar áudio completo, precisaria receber o array de áudio
        # Por enquanto, apenas detectar picos anômalos
        if audio_peak > 0.8:  # Pico muito alto
            anomaly_detected = True
            anomaly_details['audio'] = {
                'peak': audio_peak,
                'reason': 'Peak amplitude exceeds threshold'
            }

        # ====================================================================
        # CORRELAÇÃO
        # ====================================================================
        correlation_score = 0.0

        if anomaly_details.get('magnetic'):
            correlation_score += settings.CORRELATION_WEIGHT_MAGNETIC

        if anomaly_details.get('audio'):
            correlation_score += settings.CORRELATION_WEIGHT_AUDIO

        if humanoid_detected:
            correlation_score += settings.CORRELATION_WEIGHT_HUMANOID
            anomaly_details['humanoid'] = True

        # ====================================================================
        # EVENTO DETECTADO
        # ====================================================================
        if anomaly_detected and correlation_score >= settings.CORRELATION_MIN_SCORE:
            self.stats['total_events_detected'] += 1

            event_id = f"event_{int(packet['timestamp'] / 1e9)}"

            logger.warning(f"🚨 ANOMALIA DETECTADA - {client_id} - {event_id}")
            logger.warning(f"   Correlation Score: {correlation_score:.2f}")
            logger.warning(f"   Details: {anomaly_details}")

            # Notificar cliente
            await self.send_message(client_id, {
                'type': 'anomaly_detected',
                'event_id': event_id,
                'timestamp': packet['timestamp'],
                'correlation_score': correlation_score,
                'details': anomaly_details
            })

        # Enviar ACK
        await self.send_message(client_id, {
            'type': 'ack',
            'timestamp': packet['timestamp'],
            'status': 'ok'
        })


# Instância global do manager
manager = ConnectionManager()


# ============================================================================
# CICLO DE VIDA DA APLICAÇÃO
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia inicialização e shutdown"""

    # ========================================================================
    # STARTUP
    # ========================================================================
    logger.info("🚀 Iniciando Spectral Server...")

    # Validar configurações
    validate_settings()

    # Setup de logging
    logger.remove()  # Remover handler padrão
    logger.add(
        sink=settings.LOG_DIR / settings.LOG_FILE,
        rotation=settings.LOG_ROTATION,
        retention=settings.LOG_RETENTION,
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    )
    logger.add(
        sink=lambda msg: print(msg, end=''),
        level=settings.LOG_LEVEL,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )

    logger.info(f"✅ Logging configurado: {settings.LOG_LEVEL}")

    # Inicializar banco de dados (se disponível)
    if InfluxDBHandler:
        try:
            app.state.influxdb = InfluxDBHandler(
                url=settings.INFLUXDB_URL,
                token=settings.INFLUXDB_TOKEN,
                org=settings.INFLUXDB_ORG,
                bucket=settings.INFLUXDB_BUCKET
            )
            logger.info("✅ InfluxDB conectado")
        except Exception as e:
            logger.warning(f"⚠️  InfluxDB não disponível: {e}")
            app.state.influxdb = None
    else:
        app.state.influxdb = None

    if PostgresHandler:
        try:
            app.state.postgres = PostgresHandler(url=settings.POSTGRES_URL)
            logger.info("✅ PostgreSQL conectado")
        except Exception as e:
            logger.warning(f"⚠️  PostgreSQL não disponível: {e}")
            app.state.postgres = None
    else:
        app.state.postgres = None

    logger.success("✅ Spectral Server pronto!")
    logger.info(f"   🌐 Servidor: http://{settings.SERVER_HOST}:{settings.SERVER_PORT}")
    logger.info(f"   📊 Docs: http://{settings.SERVER_HOST}:{settings.SERVER_PORT}/docs")

    yield

    # ========================================================================
    # SHUTDOWN
    # ========================================================================
    logger.info("🛑 Desligando Spectral Server...")

    # Fechar conexões
    for client_id in list(manager.active_connections.keys()):
        try:
            await manager.send_message(client_id, {
                'type': 'server_shutdown',
                'message': 'Servidor sendo desligado'
            })
        except:
            pass

    logger.success("✅ Shutdown completo")


# ============================================================================
# APLICAÇÃO FASTAPI
# ============================================================================

app = FastAPI(
    title="Spectral Backend API",
    description="Sistema de detecção de anomalias ambientais em tempo real",
    version="1.0.0",
    lifespan=lifespan
)

# ============================================================================
# MIDDLEWARE
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especificar domínios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    Endpoint WebSocket principal para receber dados dos clientes
    """
    await manager.connect(websocket, client_id)

    try:
        while True:
            # Receber pacote JSON
            data = await websocket.receive_json()

            # Processar pacote
            await manager.process_sensor_packet(client_id, data)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Erro no WebSocket para {client_id}: {e}")
        manager.disconnect(client_id)


# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "name": "Spectral Backend API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
        "websocket": "/ws/{client_id}"
    }


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "clients_online": manager.stats['clients_online'],
        "total_packets": manager.stats['total_packets_received'],
        "total_events": manager.stats['total_events_detected']
    }


@app.get("/stats")
async def get_stats():
    """Estatísticas do servidor"""
    return {
        "stats": manager.stats,
        "clients": list(manager.active_connections.keys())
    }


@app.post("/test/anomaly")
async def test_anomaly_detection(data: dict):
    """
    Endpoint de teste para detecção de anomalia

    Body:
    {
        "magnitude": 75.5,
        "audio_peak": 0.9,
        "humanoid_detected": false
    }
    """
    # Simular processamento
    magnitude = data.get('magnitude', 50.0)
    audio_peak = data.get('audio_peak', 0.0)
    humanoid = data.get('humanoid_detected', False)

    # Simular Kalman
    kalman = KalmanFilter1D(
        process_variance=settings.KALMAN_PROCESS_VARIANCE,
        measurement_variance=settings.KALMAN_MEASUREMENT_VARIANCE
    )
    filtered = kalman.process(magnitude)

    # Simular detecção
    import numpy as np
    buffer = [filtered] + [50 + np.random.randn() * 5 for _ in range(99)]
    mean = np.mean(buffer)
    std = np.std(buffer)
    threshold = mean + (settings.MAGNETIC_SIGMA_MULTIPLIER * std)

    anomaly = filtered > threshold

    return {
        "input": {
            "raw_magnitude": magnitude,
            "audio_peak": audio_peak,
            "humanoid_detected": humanoid
        },
        "processing": {
            "filtered_magnitude": filtered,
            "mean": float(mean),
            "std": float(std),
            "threshold": float(threshold)
        },
        "result": {
            "anomaly_detected": anomaly,
            "z_score": float((filtered - mean) / std) if std > 0 else 0
        }
    }


# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handler global de exceções"""
    logger.error(f"Erro não tratado: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Função principal"""
    uvicorn.run(
        "main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
