"""
Sistema de Logging Estruturado

Logs em formato JSON para análise e agregação
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from enum import Enum
import traceback
import socket

from loguru import logger


# ============================================================================
# ENUMS
# ============================================================================

class LogLevel(str, Enum):
    """Níveis de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(str, Enum):
    """Categorias de log"""
    SYSTEM = "system"
    WEBSOCKET = "websocket"
    PROCESSING = "processing"
    DETECTION = "detection"
    ML_INFERENCE = "ml_inference"
    VALIDATION = "validation"
    DATABASE = "database"
    PERFORMANCE = "performance"
    SECURITY = "security"


# ============================================================================
# STRUCTURED LOGGER
# ============================================================================

class StructuredLogger:
    """
    Logger estruturado que emite logs em formato JSON

    Campos padrão:
    - timestamp: ISO 8601 timestamp
    - level: nível do log
    - category: categoria do evento
    - message: mensagem principal
    - context: dados adicionais
    - hostname: hostname do servidor
    - pid: process ID
    """

    def __init__(self, app_name: str = "spectral"):
        self.app_name = app_name
        self.hostname = socket.gethostname()
        self.pid = None

        # Inicializar PID
        import os
        self.pid = os.getpid()

    def _format_log(
        self,
        level: LogLevel,
        category: LogCategory,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ) -> Dict[str, Any]:
        """Formata log estruturado"""

        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": level.value,
            "category": category.value,
            "message": message,
            "app": self.app_name,
            "hostname": self.hostname,
            "pid": self.pid
        }

        # Adicionar contexto
        if context:
            log_entry["context"] = context

        # Adicionar exceção
        if exception:
            log_entry["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }

        return log_entry

    def debug(
        self,
        category: LogCategory,
        message: str,
        context: Optional[Dict] = None
    ):
        """Log DEBUG"""
        log_entry = self._format_log(LogLevel.DEBUG, category, message, context)
        logger.debug(json.dumps(log_entry))

    def info(
        self,
        category: LogCategory,
        message: str,
        context: Optional[Dict] = None
    ):
        """Log INFO"""
        log_entry = self._format_log(LogLevel.INFO, category, message, context)
        logger.info(json.dumps(log_entry))

    def warning(
        self,
        category: LogCategory,
        message: str,
        context: Optional[Dict] = None
    ):
        """Log WARNING"""
        log_entry = self._format_log(LogLevel.WARNING, category, message, context)
        logger.warning(json.dumps(log_entry))

    def error(
        self,
        category: LogCategory,
        message: str,
        context: Optional[Dict] = None,
        exception: Optional[Exception] = None
    ):
        """Log ERROR"""
        log_entry = self._format_log(LogLevel.ERROR, category, message, context, exception)
        logger.error(json.dumps(log_entry))

    def critical(
        self,
        category: LogCategory,
        message: str,
        context: Optional[Dict] = None,
        exception: Optional[Exception] = None
    ):
        """Log CRITICAL"""
        log_entry = self._format_log(LogLevel.CRITICAL, category, message, context, exception)
        logger.critical(json.dumps(log_entry))


# ============================================================================
# EVENT LOGGERS
# ============================================================================

class EventLogger:
    """Logger especializado para eventos específicos"""

    def __init__(self, structured_logger: StructuredLogger):
        self.logger = structured_logger

    def log_websocket_connection(
        self,
        client_id: str,
        remote_addr: str,
        user_agent: Optional[str] = None
    ):
        """Log de conexão WebSocket"""
        self.logger.info(
            category=LogCategory.WEBSOCKET,
            message=f"Cliente conectado: {client_id}",
            context={
                "client_id": client_id,
                "remote_addr": remote_addr,
                "user_agent": user_agent,
                "event": "connection"
            }
        )

    def log_websocket_disconnection(
        self,
        client_id: str,
        reason: str = "normal",
        packets_processed: int = 0
    ):
        """Log de desconexão WebSocket"""
        self.logger.info(
            category=LogCategory.WEBSOCKET,
            message=f"Cliente desconectado: {client_id}",
            context={
                "client_id": client_id,
                "reason": reason,
                "packets_processed": packets_processed,
                "event": "disconnection"
            }
        )

    def log_packet_received(
        self,
        client_id: str,
        timestamp: float,
        sensors: list,
        packet_size_bytes: int
    ):
        """Log de pacote recebido"""
        self.logger.debug(
            category=LogCategory.PROCESSING,
            message=f"Pacote recebido de {client_id}",
            context={
                "client_id": client_id,
                "timestamp": timestamp,
                "sensors": sensors,
                "packet_size_bytes": packet_size_bytes,
                "event": "packet_received"
            }
        )

    def log_processing_result(
        self,
        client_id: str,
        duration_ms: float,
        validation_passed: bool,
        quality_score: float,
        anomaly_detected: bool,
        anomaly_score: float,
        confidence: float
    ):
        """Log de resultado de processamento"""
        self.logger.info(
            category=LogCategory.PROCESSING,
            message=f"Pacote processado: {client_id}",
            context={
                "client_id": client_id,
                "duration_ms": duration_ms,
                "validation_passed": validation_passed,
                "quality_score": quality_score,
                "anomaly_detected": anomaly_detected,
                "anomaly_score": anomaly_score,
                "confidence": confidence,
                "event": "processing_complete"
            }
        )

    def log_anomaly_detected(
        self,
        client_id: str,
        timestamp: float,
        anomaly_type: str,
        score: float,
        confidence: float,
        details: Dict
    ):
        """Log de anomalia detectada"""
        self.logger.warning(
            category=LogCategory.DETECTION,
            message=f"ANOMALIA DETECTADA - {client_id}",
            context={
                "client_id": client_id,
                "timestamp": timestamp,
                "anomaly_type": anomaly_type,
                "score": score,
                "confidence": confidence,
                "details": details,
                "event": "anomaly_detected",
                "severity": "high" if confidence > 0.8 else "medium"
            }
        )

    def log_ml_inference(
        self,
        model_name: str,
        client_id: str,
        input_shape: tuple,
        output_shape: tuple,
        duration_ms: float,
        prediction: Any
    ):
        """Log de inferência de ML"""
        self.logger.info(
            category=LogCategory.ML_INFERENCE,
            message=f"Inferência ML: {model_name}",
            context={
                "model_name": model_name,
                "client_id": client_id,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "duration_ms": duration_ms,
                "prediction": str(prediction),
                "event": "ml_inference"
            }
        )

    def log_validation_error(
        self,
        client_id: str,
        error_type: str,
        field: str,
        message: str,
        severity: str
    ):
        """Log de erro de validação"""
        self.logger.warning(
            category=LogCategory.VALIDATION,
            message=f"Erro de validação: {client_id}",
            context={
                "client_id": client_id,
                "error_type": error_type,
                "field": field,
                "validation_message": message,
                "severity": severity,
                "event": "validation_error"
            }
        )

    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        threshold: Optional[float] = None
    ):
        """Log de métrica de performance"""
        exceeded = threshold is not None and value > threshold

        self.logger.info(
            category=LogCategory.PERFORMANCE,
            message=f"Métrica: {metric_name} = {value}{unit}",
            context={
                "metric_name": metric_name,
                "value": value,
                "unit": unit,
                "threshold": threshold,
                "threshold_exceeded": exceeded,
                "event": "performance_metric"
            }
        )

    def log_database_operation(
        self,
        operation: str,
        database: str,
        collection: str,
        duration_ms: float,
        success: bool,
        records_affected: int = 0
    ):
        """Log de operação de banco de dados"""
        level = LogLevel.INFO if success else LogLevel.ERROR

        self.logger.info(
            category=LogCategory.DATABASE,
            message=f"Operação DB: {operation} em {database}.{collection}",
            context={
                "operation": operation,
                "database": database,
                "collection": collection,
                "duration_ms": duration_ms,
                "success": success,
                "records_affected": records_affected,
                "event": "database_operation"
            }
        )

    def log_security_event(
        self,
        event_type: str,
        client_id: str,
        remote_addr: str,
        description: str,
        severity: str = "medium"
    ):
        """Log de evento de segurança"""
        self.logger.warning(
            category=LogCategory.SECURITY,
            message=f"Evento de segurança: {event_type}",
            context={
                "event_type": event_type,
                "client_id": client_id,
                "remote_addr": remote_addr,
                "description": description,
                "severity": severity,
                "event": "security_event"
            }
        )


# ============================================================================
# SETUP
# ============================================================================

def setup_structured_logging(
    log_dir: Path,
    app_name: str = "spectral",
    level: str = "INFO",
    rotation: str = "100 MB",
    retention: str = "30 days"
) -> tuple[StructuredLogger, EventLogger]:
    """
    Configura sistema de logging estruturado

    Returns:
        (StructuredLogger, EventLogger)
    """

    # Criar diretório de logs
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Remover handlers padrão
    logger.remove()

    # Arquivo JSON (estruturado)
    logger.add(
        sink=log_dir / f"{app_name}.json",
        format="{message}",  # Apenas a mensagem (já formatada como JSON)
        rotation=rotation,
        retention=retention,
        level=level,
        serialize=False  # Não serializar novamente
    )

    # Arquivo de texto (legível por humanos)
    logger.add(
        sink=log_dir / f"{app_name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        level=level,
        colorize=False
    )

    # Console (colorido)
    logger.add(
        sink=sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level=level,
        colorize=True
    )

    # Criar loggers
    structured_logger = StructuredLogger(app_name=app_name)
    event_logger = EventLogger(structured_logger)

    return structured_logger, event_logger


# ============================================================================
# LOG AGGREGATOR
# ============================================================================

class LogAggregator:
    """
    Agrega logs para análise

    Lê arquivo JSON e extrai métricas
    """

    def __init__(self, log_file: Path):
        self.log_file = log_file

    def get_recent_logs(
        self,
        limit: int = 100,
        category: Optional[LogCategory] = None,
        level: Optional[LogLevel] = None
    ) -> list[Dict]:
        """Retorna logs recentes"""

        logs = []

        try:
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line)

                        # Filtrar por categoria
                        if category and log_entry.get('category') != category.value:
                            continue

                        # Filtrar por level
                        if level and log_entry.get('level') != level.value:
                            continue

                        logs.append(log_entry)

                    except json.JSONDecodeError:
                        continue

        except FileNotFoundError:
            return []

        # Retornar últimos N
        return logs[-limit:]

    def get_anomaly_timeline(self, hours: int = 24) -> list[Dict]:
        """Retorna timeline de anomalias"""

        cutoff_time = datetime.utcnow().timestamp() - (hours * 3600)
        anomalies = []

        for log in self.get_recent_logs(limit=10000, category=LogCategory.DETECTION):
            timestamp = datetime.fromisoformat(log['timestamp'].replace('Z', ''))

            if timestamp.timestamp() > cutoff_time:
                anomalies.append(log)

        return anomalies

    def get_error_summary(self) -> Dict:
        """Retorna resumo de erros"""

        errors = self.get_recent_logs(limit=1000, level=LogLevel.ERROR)

        error_counts = {}
        for error in errors:
            exception_type = error.get('exception', {}).get('type', 'Unknown')
            error_counts[exception_type] = error_counts.get(exception_type, 0) + 1

        return {
            'total_errors': len(errors),
            'error_types': error_counts,
            'recent_errors': errors[-10:]
        }


# ============================================================================
# TESTE
# ============================================================================

if __name__ == "__main__":
    # Setup
    structured, events = setup_structured_logging(
        log_dir=Path("./logs"),
        app_name="spectral_test",
        level="DEBUG"
    )

    # Testar eventos
    events.log_websocket_connection(
        client_id="test_client_1",
        remote_addr="192.168.1.100",
        user_agent="SpectralAndroid/1.0"
    )

    events.log_anomaly_detected(
        client_id="test_client_1",
        timestamp=time.time(),
        anomaly_type="magnetic",
        score=0.85,
        confidence=0.92,
        details={"sensor": "magnetometer", "threshold": 75.0}
    )

    events.log_websocket_disconnection(
        client_id="test_client_1",
        reason="normal",
        packets_processed=150
    )

    print("✅ Logs estruturados criados em ./logs/")
