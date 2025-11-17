"""
Sistema de Métricas e Monitoramento

Integração com Prometheus para métricas de produção
"""

from typing import Dict, Optional
from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, generate_latest,
    CONTENT_TYPE_LATEST
)
import time
from functools import wraps
import psutil
import asyncio


# ============================================================================
# REGISTRY
# ============================================================================

# Registry customizado para evitar conflitos
registry = CollectorRegistry()


# ============================================================================
# MÉTRICAS DE REQUISIÇÕES
# ============================================================================

# Contador de requisições WebSocket
websocket_connections_total = Counter(
    'spectral_websocket_connections_total',
    'Total de conexões WebSocket',
    ['client_id'],
    registry=registry
)

websocket_disconnections_total = Counter(
    'spectral_websocket_disconnections_total',
    'Total de desconexões WebSocket',
    ['client_id', 'reason'],
    registry=registry
)

# Gauge de conexões ativas
websocket_connections_active = Gauge(
    'spectral_websocket_connections_active',
    'Conexões WebSocket ativas',
    registry=registry
)


# ============================================================================
# MÉTRICAS DE PROCESSAMENTO
# ============================================================================

# Contador de pacotes
packets_received_total = Counter(
    'spectral_packets_received_total',
    'Total de pacotes de sensores recebidos',
    ['client_id'],
    registry=registry
)

packets_processed_total = Counter(
    'spectral_packets_processed_total',
    'Total de pacotes processados com sucesso',
    ['client_id'],
    registry=registry
)

packets_failed_total = Counter(
    'spectral_packets_failed_total',
    'Total de pacotes com erro no processamento',
    ['client_id', 'error_type'],
    registry=registry
)

# Histograma de latência de processamento
processing_duration_seconds = Histogram(
    'spectral_processing_duration_seconds',
    'Duração do processamento de pacotes',
    ['client_id'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
    registry=registry
)


# ============================================================================
# MÉTRICAS DE DETECÇÃO
# ============================================================================

# Contador de anomalias
anomalies_detected_total = Counter(
    'spectral_anomalies_detected_total',
    'Total de anomalias detectadas',
    ['client_id', 'anomaly_type'],
    registry=registry
)

# Gauge de score de anomalia
anomaly_score_current = Gauge(
    'spectral_anomaly_score_current',
    'Score atual de anomalia',
    ['client_id'],
    registry=registry
)

# Histograma de confiança
confidence_score = Histogram(
    'spectral_confidence_score',
    'Score de confiança da detecção',
    ['client_id'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=registry
)


# ============================================================================
# MÉTRICAS DE QUALIDADE
# ============================================================================

# Gauge de qualidade de dados
data_quality_score = Gauge(
    'spectral_data_quality_score',
    'Score de qualidade dos dados',
    ['client_id', 'sensor'],
    registry=registry
)

# Gauge de SNR
signal_to_noise_ratio = Gauge(
    'spectral_signal_to_noise_ratio_db',
    'Signal-to-Noise Ratio em dB',
    ['client_id', 'sensor'],
    registry=registry
)

# Contador de erros de validação
validation_errors_total = Counter(
    'spectral_validation_errors_total',
    'Total de erros de validação',
    ['client_id', 'error_type'],
    registry=registry
)


# ============================================================================
# MÉTRICAS DE SISTEMA
# ============================================================================

# CPU usage
cpu_usage_percent = Gauge(
    'spectral_cpu_usage_percent',
    'Uso de CPU do processo',
    registry=registry
)

# Memory usage
memory_usage_bytes = Gauge(
    'spectral_memory_usage_bytes',
    'Uso de memória do processo',
    registry=registry
)

# GPU metrics (se disponível)
gpu_usage_percent = Gauge(
    'spectral_gpu_usage_percent',
    'Uso de GPU',
    ['gpu_id'],
    registry=registry
)

gpu_memory_usage_bytes = Gauge(
    'spectral_gpu_memory_usage_bytes',
    'Uso de memória GPU',
    ['gpu_id'],
    registry=registry
)


# ============================================================================
# MÉTRICAS DE ML
# ============================================================================

# Contador de inferências
ml_inferences_total = Counter(
    'spectral_ml_inferences_total',
    'Total de inferências de ML',
    ['model_name', 'client_id'],
    registry=registry
)

# Histograma de latência de inferência
ml_inference_duration_seconds = Histogram(
    'spectral_ml_inference_duration_seconds',
    'Duração da inferência de ML',
    ['model_name'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=registry
)


# ============================================================================
# COLETOR DE MÉTRICAS DE SISTEMA
# ============================================================================

class SystemMetricsCollector:
    """Coleta métricas do sistema periodicamente"""

    def __init__(self, interval: float = 5.0):
        self.interval = interval
        self.process = psutil.Process()
        self._running = False
        self._task = None

    async def start(self):
        """Inicia coleta"""
        self._running = True
        self._task = asyncio.create_task(self._collect_loop())

    async def stop(self):
        """Para coleta"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _collect_loop(self):
        """Loop de coleta"""
        while self._running:
            try:
                self._collect_metrics()
                await asyncio.sleep(self.interval)
            except Exception as e:
                print(f"Erro ao coletar métricas: {e}")

    def _collect_metrics(self):
        """Coleta métricas do sistema"""

        # CPU
        cpu_percent = self.process.cpu_percent(interval=0.1)
        cpu_usage_percent.set(cpu_percent)

        # Memory
        mem_info = self.process.memory_info()
        memory_usage_bytes.set(mem_info.rss)

        # GPU (se disponível)
        try:
            import pynvml
            pynvml.nvmlInit()

            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_usage_percent.labels(gpu_id=str(i)).set(util.gpu)

                # Memory
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory_usage_bytes.labels(gpu_id=str(i)).set(mem.used)

            pynvml.nvmlShutdown()

        except Exception:
            # GPU não disponível ou erro
            pass


# ============================================================================
# DECORATORS
# ============================================================================

def track_processing_time(client_id: str = "unknown"):
    """Decorator para rastrear tempo de processamento"""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                packets_processed_total.labels(client_id=client_id).inc()
                return result
            except Exception as e:
                packets_failed_total.labels(
                    client_id=client_id,
                    error_type=type(e).__name__
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                processing_duration_seconds.labels(client_id=client_id).observe(duration)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                packets_processed_total.labels(client_id=client_id).inc()
                return result
            except Exception as e:
                packets_failed_total.labels(
                    client_id=client_id,
                    error_type=type(e).__name__
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                processing_duration_seconds.labels(client_id=client_id).observe(duration)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def track_ml_inference(model_name: str):
    """Decorator para rastrear inferências de ML"""

    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                ml_inference_duration_seconds.labels(model_name=model_name).observe(duration)
                ml_inferences_total.labels(
                    model_name=model_name,
                    client_id=kwargs.get('client_id', 'unknown')
                ).inc()

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                ml_inference_duration_seconds.labels(model_name=model_name).observe(duration)
                ml_inferences_total.labels(
                    model_name=model_name,
                    client_id=kwargs.get('client_id', 'unknown')
                ).inc()

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# ============================================================================
# HELPERS
# ============================================================================

def record_anomaly(client_id: str, anomaly_type: str, score: float, confidence: float):
    """Registra anomalia detectada"""
    anomalies_detected_total.labels(
        client_id=client_id,
        anomaly_type=anomaly_type
    ).inc()

    anomaly_score_current.labels(client_id=client_id).set(score)
    confidence_score.labels(client_id=client_id).observe(confidence)


def record_quality(client_id: str, sensor: str, quality: float, snr: float):
    """Registra qualidade de dados"""
    data_quality_score.labels(client_id=client_id, sensor=sensor).set(quality)
    signal_to_noise_ratio.labels(client_id=client_id, sensor=sensor).set(snr)


def record_validation_error(client_id: str, error_type: str):
    """Registra erro de validação"""
    validation_errors_total.labels(
        client_id=client_id,
        error_type=error_type
    ).inc()


def get_metrics() -> bytes:
    """Retorna métricas no formato Prometheus"""
    return generate_latest(registry)


def get_content_type() -> str:
    """Retorna content type das métricas"""
    return CONTENT_TYPE_LATEST
