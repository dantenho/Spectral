"""
Módulo de Monitoramento e Métricas

Exports:
- Prometheus metrics
- Structured logging
- System metrics collector
"""

from .metrics import (
    # Métricas
    websocket_connections_total,
    packets_received_total,
    anomalies_detected_total,
    processing_duration_seconds,
    confidence_score,
    data_quality_score,
    # Helpers
    record_anomaly,
    record_quality,
    record_validation_error,
    get_metrics,
    get_content_type,
    # Collector
    SystemMetricsCollector,
    # Decorators
    track_processing_time,
    track_ml_inference
)

from .structured_logging import (
    StructuredLogger,
    EventLogger,
    LogCategory,
    LogLevel,
    setup_structured_logging,
    LogAggregator
)

__all__ = [
    # Metrics
    'websocket_connections_total',
    'packets_received_total',
    'anomalies_detected_total',
    'processing_duration_seconds',
    'confidence_score',
    'data_quality_score',
    'record_anomaly',
    'record_quality',
    'record_validation_error',
    'get_metrics',
    'get_content_type',
    'SystemMetricsCollector',
    'track_processing_time',
    'track_ml_inference',

    # Logging
    'StructuredLogger',
    'EventLogger',
    'LogCategory',
    'LogLevel',
    'setup_structured_logging',
    'LogAggregator',
]
