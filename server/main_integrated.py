"""
Servidor Principal Integrado do Spectral

Sistema completo de detecção de anomalias com todos os módulos integrados:
- Filtros de Kalman
- Algoritmos avançados de precisão
- Métricas de qualidade
- Análise estatística
- Validação de dados
- Redes neurais (quando treinadas)
"""

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import deque

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# Configurações
from config.settings import settings, validate_settings

# Core - Kalman Filters
from core.kalman_filter import (
    KalmanFilter1D, KalmanFilter3D,
    AdaptiveKalmanFilter, ExtendedKalmanFilter
)

# Processing - Audio
from processing.audio_variants import AudioEnsemble

# Processing - Advanced Algorithms
from processing.advanced_algorithms import (
    CUSUMDetector, EWMADetector,
    CrossCorrelationAnalyzer, PowerSpectralDensityAnalyzer,
    BayesianClassifier, MahalanobisDetector,
    EnsembleConfidenceCalculator
)

# Processing - Quality Metrics
from processing.quality_metrics import (
    DataQualityAnalyzer, StatisticalQualityMetrics,
    MultiSensorQualityAnalyzer
)

# Processing - Statistical Analysis
from processing.statistical_analysis import (
    DistributionAnalyzer, TimeSeriesAnalyzer,
    ChangePointDetector, ExtremeValueAnalyzer,
    MultivariateAnalyzer
)

# Processing - Data Validation
from processing.data_validation import (
    SensorDataValidator, SchemaValidator,
    ConsistencyValidator, BatchValidator,
    ValidationSeverity
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SensorData:
    """Estrutura de dados de sensor"""
    timestamp: float
    x: float
    y: float
    z: float
    magnitude: Optional[float] = None

    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])


@dataclass
class ProcessingResult:
    """Resultado do processamento"""
    timestamp: float
    client_id: str

    # Dados brutos
    raw_data: Dict

    # Validação
    validation_passed: bool
    validation_errors: int
    validation_warnings: int

    # Qualidade
    overall_quality: float
    snr: float
    stability: float

    # Filtros de Kalman
    kalman_filtered: Dict[str, np.ndarray]

    # Detecção de anomalias
    anomaly_detected: bool
    anomaly_score: float
    anomaly_details: Dict

    # Análise estatística
    statistical_metrics: Dict

    # Confiança
    confidence: float


# ============================================================================
# PROCESSADOR INTEGRADO
# ============================================================================

class IntegratedProcessor:
    """
    Processador integrado que combina todos os módulos de análise
    """

    def __init__(self, client_id: str):
        self.client_id = client_id

        # Kalman Filters por sensor
        self.kalman_filters = {
            'accelerometer': KalmanFilter3D(
                process_variance=1e-5,
                measurement_variance=1e-2
            ),
            'gyroscope': KalmanFilter3D(
                process_variance=1e-5,
                measurement_variance=1e-2
            ),
            'magnetometer': AdaptiveKalmanFilter(
                process_variance=1e-5,
                initial_measurement_variance=1e-2,
                adaptation_rate=0.1
            )
        }

        # Detectores de anomalia
        self.cusum_detectors = {
            sensor: CUSUMDetector(target=0.0, k=0.5, h=5.0)
            for sensor in ['accelerometer', 'gyroscope', 'magnetometer']
        }

        self.ewma_detectors = {
            sensor: EWMADetector(lambda_=0.2, L=3.0)
            for sensor in ['accelerometer', 'gyroscope', 'magnetometer']
        }

        # Mahalanobis detector (multivariado)
        self.mahalanobis = MahalanobisDetector(threshold=3.0)
        self.mahalanobis_samples = []

        # Bayesian classifier
        self.bayesian = BayesianClassifier(
            classes=['Normal', 'Anomalia', 'Interferência', 'EVP']
        )

        # Analisadores
        self.quality_analyzer = DataQualityAnalyzer(sample_rate=100.0)
        self.correlation_analyzer = CrossCorrelationAnalyzer(max_lag=50)
        self.psd_analyzer = PowerSpectralDensityAnalyzer(sample_rate=100.0)
        self.ts_analyzer = TimeSeriesAnalyzer()
        self.cp_detector = ChangePointDetector()

        # Validadores
        self.sensor_validator = SensorDataValidator()
        self.consistency_validator = ConsistencyValidator()

        # Confidence calculator
        self.confidence_calc = EnsembleConfidenceCalculator()

        # Buffers de dados
        self.sensor_buffers = {
            'accelerometer': deque(maxlen=1000),
            'gyroscope': deque(maxlen=1000),
            'magnetometer': deque(maxlen=1000)
        }

        # Buffer para detecção multivariada
        self.multivariate_buffer = deque(maxlen=100)

        # Estatísticas
        self.packet_count = 0
        self.anomaly_count = 0

        logger.info(f"✅ Processador integrado criado para {client_id}")

    async def process_packet(self, packet: Dict) -> ProcessingResult:
        """
        Processa pacote completo com todos os módulos integrados

        Pipeline:
        1. Validação de dados
        2. Filtragem de Kalman
        3. Análise de qualidade
        4. Detecção de anomalias (CUSUM, EWMA, Mahalanobis, Bayesian)
        5. Análise estatística
        6. Cálculo de confiança
        """

        self.packet_count += 1
        timestamp = packet.get('timestamp', time.time())

        # ====================================================================
        # ETAPA 1: VALIDAÇÃO
        # ====================================================================

        validation_result = self.sensor_validator.validate_sensor_packet(packet)

        if not validation_result.is_valid:
            logger.warning(
                f"⚠️  Pacote inválido de {self.client_id}: "
                f"{validation_result.errors} erros, {validation_result.warnings} avisos"
            )

            # Se crítico, rejeitar
            if validation_result.critical > 0:
                raise ValueError("Pacote com erros críticos")

        # Validação de consistência
        consistency_result = self.consistency_validator.validate_cross_field_consistency(packet)

        # ====================================================================
        # ETAPA 2: EXTRAÇÃO E FILTRAGEM DE KALMAN
        # ====================================================================

        sensors_data = packet.get('sensors', {})
        kalman_filtered = {}

        for sensor_name in ['accelerometer', 'gyroscope', 'magnetometer']:
            if sensor_name in sensors_data:
                sensor_data = sensors_data[sensor_name]

                # Processar com Kalman
                if sensor_name == 'magnetometer':
                    # Scalar Kalman para magnetômetro
                    magnitude = sensor_data.get('magnitude', 0.0)
                    filtered = self.kalman_filters[sensor_name].process(magnitude)
                    kalman_filtered[sensor_name] = filtered

                    # Buffer
                    self.sensor_buffers[sensor_name].append(filtered)

                else:
                    # 3D Kalman para accel/gyro
                    x = sensor_data.get('x', 0.0)
                    y = sensor_data.get('y', 0.0)
                    z = sensor_data.get('z', 0.0)

                    measurement = np.array([x, y, z])
                    filtered = self.kalman_filters[sensor_name].process(measurement)
                    kalman_filtered[sensor_name] = filtered

                    # Buffer (magnitude)
                    magnitude = np.linalg.norm(filtered)
                    self.sensor_buffers[sensor_name].append(magnitude)

        # ====================================================================
        # ETAPA 3: ANÁLISE DE QUALIDADE
        # ====================================================================

        overall_quality = 1.0
        snr = 0.0
        stability = 1.0

        # Analisar qualidade de cada sensor
        quality_reports = {}

        for sensor_name, buffer in self.sensor_buffers.items():
            if len(buffer) >= 100:
                data = np.array(list(buffer))
                report = self.quality_analyzer.analyze(data)
                quality_reports[sensor_name] = report

                # Média de qualidade
                overall_quality = min(overall_quality, report.overall_score)
                snr = max(snr, report.snr)
                stability = min(stability, report.stability_score)

        # Qualidade multi-sensor
        sensor_arrays = {
            name: np.array(list(buffer))
            for name, buffer in self.sensor_buffers.items()
            if len(buffer) >= 50
        }

        if len(sensor_arrays) >= 2:
            correlation_quality = MultiSensorQualityAnalyzer.analyze_correlation_quality(
                sensor_arrays
            )
            overall_quality = (overall_quality + correlation_quality['quality']) / 2

        # ====================================================================
        # ETAPA 4: DETECÇÃO DE ANOMALIAS
        # ====================================================================

        anomaly_detected = False
        anomaly_score = 0.0
        anomaly_details = {}

        # 4A: CUSUM Detection
        cusum_detections = {}
        for sensor_name, filtered in kalman_filtered.items():
            if isinstance(filtered, (int, float)):
                value = filtered
            else:
                value = np.linalg.norm(filtered)

            cusum_result = self.cusum_detectors[sensor_name].update(value)
            cusum_detections[sensor_name] = cusum_result

            if cusum_result['change_detected']:
                anomaly_detected = True
                anomaly_score += 0.3
                anomaly_details[f'cusum_{sensor_name}'] = cusum_result

        # 4B: EWMA Detection
        ewma_detections = {}
        for sensor_name, filtered in kalman_filtered.items():
            if isinstance(filtered, (int, float)):
                value = filtered
            else:
                value = np.linalg.norm(filtered)

            ewma_result = self.ewma_detectors[sensor_name].update(value)
            ewma_detections[sensor_name] = ewma_result

            if ewma_result['out_of_control']:
                anomaly_detected = True
                anomaly_score += 0.2
                anomaly_details[f'ewma_{sensor_name}'] = ewma_result

        # 4C: Mahalanobis Detection (multivariate)
        if len(kalman_filtered) >= 2:
            # Criar vetor de features
            feature_vector = []
            for sensor_name in ['accelerometer', 'gyroscope', 'magnetometer']:
                if sensor_name in kalman_filtered:
                    val = kalman_filtered[sensor_name]
                    if isinstance(val, (int, float)):
                        feature_vector.append(val)
                    else:
                        feature_vector.extend(val)

            feature_vector = np.array(feature_vector)
            self.multivariate_buffer.append(feature_vector)

            # Treinar Mahalanobis se tiver amostras
            if len(self.multivariate_buffer) >= 50:
                samples = np.array(list(self.multivariate_buffer))

                # Re-fit periodicamente
                if self.packet_count % 100 == 0:
                    self.mahalanobis.fit(samples[-50:])

                # Detectar
                if self.mahalanobis.mean is not None:
                    mahal_result = self.mahalanobis.detect(feature_vector)

                    if mahal_result['is_anomaly']:
                        anomaly_detected = True
                        anomaly_score += 0.4
                        anomaly_details['mahalanobis'] = mahal_result

        # 4D: Bayesian Classification
        # Calcular likelihoods baseado nas detecções
        evidence = {
            'Normal': 1.0 - anomaly_score,
            'Anomalia': anomaly_score,
            'Interferência': 0.5 * anomaly_score if overall_quality < 0.5 else 0.0,
            'EVP': 0.3 * anomaly_score if overall_quality > 0.7 else 0.0
        }

        bayesian_result = self.bayesian.update(evidence)

        if bayesian_result['predicted_class'] != 'Normal':
            anomaly_details['bayesian'] = bayesian_result

        # ====================================================================
        # ETAPA 5: ANÁLISE ESTATÍSTICA
        # ====================================================================

        statistical_metrics = {}

        # Análise de tendência
        for sensor_name, buffer in self.sensor_buffers.items():
            if len(buffer) >= 50:
                data = np.array(list(buffer))

                trend_result = self.ts_analyzer.detect_trend(data)
                if trend_result['has_trend']:
                    statistical_metrics[f'{sensor_name}_trend'] = trend_result

                # ACF
                if len(buffer) >= 100:
                    acf_result = self.ts_analyzer.autocorrelation_analysis(data, max_lag=20)
                    statistical_metrics[f'{sensor_name}_acf'] = {
                        'first_significant_lag': acf_result['first_significant_lag'],
                        'max_acf': acf_result['max_acf']
                    }

        # ====================================================================
        # ETAPA 6: CÁLCULO DE CONFIANÇA
        # ====================================================================

        # Criar distribuição de probabilidades
        probabilities = np.array([
            bayesian_result['posteriors']['Normal'],
            bayesian_result['posteriors']['Anomalia'],
            bayesian_result['posteriors']['Interferência'],
            bayesian_result['posteriors']['EVP']
        ])

        confidence_metrics = self.confidence_calc.calculate_all(probabilities)
        confidence = confidence_metrics['overall_confidence']

        # Ajustar confiança pela qualidade
        confidence = confidence * overall_quality

        # ====================================================================
        # RESULTADO FINAL
        # ====================================================================

        # Normalizar anomaly_score
        anomaly_score = min(1.0, anomaly_score)

        # Se anomaly_score alto e confiança alta, confirmar anomalia
        if anomaly_score > 0.5 and confidence > 0.7:
            anomaly_detected = True
            self.anomaly_count += 1

        result = ProcessingResult(
            timestamp=timestamp,
            client_id=self.client_id,
            raw_data=packet,
            validation_passed=validation_result.is_valid,
            validation_errors=validation_result.errors,
            validation_warnings=validation_result.warnings,
            overall_quality=overall_quality,
            snr=snr,
            stability=stability,
            kalman_filtered={k: v.tolist() if isinstance(v, np.ndarray) else v
                           for k, v in kalman_filtered.items()},
            anomaly_detected=anomaly_detected,
            anomaly_score=anomaly_score,
            anomaly_details=anomaly_details,
            statistical_metrics=statistical_metrics,
            confidence=confidence
        )

        return result


# ============================================================================
# GERENCIADOR DE CONEXÕES
# ============================================================================

class ConnectionManager:
    """Gerencia conexões WebSocket com processadores integrados"""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.processors: Dict[str, IntegratedProcessor] = {}

        # Estatísticas globais
        self.stats = {
            'total_clients': 0,
            'clients_online': 0,
            'total_packets': 0,
            'total_anomalies': 0,
            'uptime_start': time.time()
        }

    async def connect(self, websocket: WebSocket, client_id: str):
        """Conecta novo cliente"""
        await websocket.accept()

        self.active_connections[client_id] = websocket
        self.processors[client_id] = IntegratedProcessor(client_id)

        self.stats['total_clients'] += 1
        self.stats['clients_online'] = len(self.active_connections)

        logger.info(f"✅ Cliente conectado: {client_id} (Total: {self.stats['clients_online']})")

        await self.send_message(client_id, {
            'type': 'connected',
            'message': f'Conectado ao Spectral Server Integrado',
            'client_id': client_id,
            'server_version': '2.0.0-integrated',
            'timestamp': time.time()
        })

    def disconnect(self, client_id: str):
        """Desconecta cliente"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]

        if client_id in self.processors:
            processor = self.processors[client_id]
            logger.info(
                f"📊 Estatísticas de {client_id}: "
                f"{processor.packet_count} pacotes, "
                f"{processor.anomaly_count} anomalias"
            )
            del self.processors[client_id]

        self.stats['clients_online'] = len(self.active_connections)

        logger.info(f"❌ Cliente desconectado: {client_id}")

    async def send_message(self, client_id: str, message: Dict):
        """Envia mensagem para cliente"""
        websocket = self.active_connections.get(client_id)
        if websocket:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Erro ao enviar mensagem para {client_id}: {e}")

    async def process_packet(self, client_id: str, packet: Dict):
        """Processa pacote usando processador integrado"""

        self.stats['total_packets'] += 1

        processor = self.processors.get(client_id)
        if not processor:
            logger.error(f"Processador não encontrado para {client_id}")
            return

        try:
            # Processar com pipeline integrado
            result = await processor.process_packet(packet)

            # Atualizar stats
            if result.anomaly_detected:
                self.stats['total_anomalies'] += 1

            # Enviar resultado de volta
            await self.send_message(client_id, {
                'type': 'processing_result',
                'timestamp': result.timestamp,
                'validation': {
                    'passed': result.validation_passed,
                    'errors': result.validation_errors,
                    'warnings': result.validation_warnings
                },
                'quality': {
                    'overall': result.overall_quality,
                    'snr': result.snr,
                    'stability': result.stability
                },
                'anomaly': {
                    'detected': result.anomaly_detected,
                    'score': result.anomaly_score,
                    'confidence': result.confidence,
                    'details': result.anomaly_details if result.anomaly_detected else {}
                },
                'statistical': result.statistical_metrics
            })

            # Se anomalia detectada, log especial
            if result.anomaly_detected:
                logger.warning(
                    f"🚨 ANOMALIA - {client_id} - "
                    f"Score: {result.anomaly_score:.2f}, "
                    f"Confiança: {result.confidence:.2f}, "
                    f"Qualidade: {result.overall_quality:.2f}"
                )

        except Exception as e:
            logger.error(f"Erro ao processar pacote de {client_id}: {e}", exc_info=True)

            await self.send_message(client_id, {
                'type': 'error',
                'message': f'Erro no processamento: {str(e)}',
                'timestamp': time.time()
            })


# Instância global
manager = ConnectionManager()


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia ciclo de vida"""

    logger.info("🚀 Iniciando Spectral Server Integrado v2.0...")

    # Validar configurações
    validate_settings()

    # Setup de logging
    logger.remove()
    logger.add(
        sink=settings.LOG_DIR / "spectral_integrated.log",
        rotation="100 MB",
        retention="30 days",
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
    )
    logger.add(
        sink=lambda msg: print(msg, end=''),
        level=settings.LOG_LEVEL,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
    )

    logger.success("✅ Spectral Server Integrado pronto!")
    logger.info(f"   🌐 Servidor: http://{settings.SERVER_HOST}:{settings.SERVER_PORT}")
    logger.info(f"   📊 Docs: http://{settings.SERVER_HOST}:{settings.SERVER_PORT}/docs")
    logger.info("   🔧 Módulos integrados:")
    logger.info("      - Filtros de Kalman (1D, 3D, Adaptativo)")
    logger.info("      - Detectores de anomalia (CUSUM, EWMA, Mahalanobis, Bayesian)")
    logger.info("      - Análise de qualidade (SNR, THD, Estabilidade)")
    logger.info("      - Análise estatística (Tendência, ACF, Change Points)")
    logger.info("      - Validação de dados (Ranges, Tipos, Consistência)")

    yield

    logger.info("🛑 Desligando servidor...")
    logger.success("✅ Shutdown completo")


app = FastAPI(
    title="Spectral Integrated Backend API",
    description="Sistema integrado de detecção de anomalias ambientais",
    version="2.0.0-integrated",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """Endpoint WebSocket principal"""
    await manager.connect(websocket, client_id)

    try:
        while True:
            data = await websocket.receive_json()
            await manager.process_packet(client_id, data)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Erro no WebSocket para {client_id}: {e}")
        manager.disconnect(client_id)


@app.get("/")
async def root():
    """Raiz"""
    return {
        "name": "Spectral Integrated Backend API",
        "version": "2.0.0-integrated",
        "status": "online",
        "modules": [
            "kalman_filters",
            "advanced_algorithms",
            "quality_metrics",
            "statistical_analysis",
            "data_validation"
        ]
    }


@app.get("/health")
async def health_check():
    """Health check"""
    uptime = time.time() - manager.stats['uptime_start']

    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "clients_online": manager.stats['clients_online'],
        "total_packets": manager.stats['total_packets'],
        "total_anomalies": manager.stats['total_anomalies'],
        "anomaly_rate": manager.stats['total_anomalies'] / max(manager.stats['total_packets'], 1)
    }


@app.get("/stats")
async def get_stats():
    """Estatísticas detalhadas"""
    processors_stats = {}

    for client_id, processor in manager.processors.items():
        processors_stats[client_id] = {
            'packets_processed': processor.packet_count,
            'anomalies_detected': processor.anomaly_count,
            'anomaly_rate': processor.anomaly_count / max(processor.packet_count, 1)
        }

    return {
        "global": manager.stats,
        "processors": processors_stats
    }


def main():
    """Entry point"""
    uvicorn.run(
        "main_integrated:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )


if __name__ == "__main__":
    main()
