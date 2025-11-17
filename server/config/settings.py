"""
Configurações do Servidor Spectral
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional


class Settings(BaseSettings):
    """Configurações carregadas de variáveis de ambiente"""

    # ========================================================================
    # SERVIDOR
    # ========================================================================
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8000
    DEBUG: bool = False
    RELOAD: bool = False

    # ========================================================================
    # INFLUXDB (Time Series)
    # ========================================================================
    INFLUXDB_URL: str = "http://localhost:8086"
    INFLUXDB_TOKEN: str = ""
    INFLUXDB_ORG: str = "spectral"
    INFLUXDB_BUCKET: str = "sensors"

    # ========================================================================
    # POSTGRESQL (Relacional)
    # ========================================================================
    POSTGRES_URL: str = "postgresql://spectral:spectral_password@localhost:5432/spectral"
    POSTGRES_USER: str = "spectral"
    POSTGRES_PASSWORD: str = "spectral_password"
    POSTGRES_DB: str = "spectral"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432

    # ========================================================================
    # STORAGE
    # ========================================================================
    EVENT_STORAGE_DIR: Path = Path("../data/events")
    MODEL_STORAGE_DIR: Path = Path("../models")
    LOG_DIR: Path = Path("../logs")

    # ========================================================================
    # ANOMALY DETECTION - Magnetômetro
    # ========================================================================
    # Kalman Filter
    KALMAN_PROCESS_VARIANCE: float = 1e-5
    KALMAN_MEASUREMENT_VARIANCE: float = 5e-2

    # Detecção de anomalia
    MAGNETIC_WINDOW_SIZE: int = 100
    MAGNETIC_SIGMA_MULTIPLIER: float = 3.0

    # ========================================================================
    # ANOMALY DETECTION - Áudio
    # ========================================================================
    AUDIO_SAMPLE_RATE: int = 44100

    # FFT
    AUDIO_INFRASOUND_THRESHOLD_DB: float = -40
    AUDIO_ULTRASOUND_THRESHOLD_DB: float = -40
    AUDIO_ANOMALY_THRESHOLD_DB: float = -20

    # STFT
    AUDIO_STFT_WINDOW: int = 2048
    AUDIO_STFT_HOP: int = 512

    # Formantes (EVP)
    AUDIO_LPC_ORDER: int = 12
    AUDIO_MIN_FORMANTS: int = 2

    # Ensemble
    AUDIO_ENSEMBLE_THRESHOLD: float = 0.5

    # ========================================================================
    # CORRELATION ENGINE
    # ========================================================================
    CORRELATION_TIME_WINDOW_MS: int = 1000
    CORRELATION_MIN_SCORE: float = 0.6

    # Pesos para scoring
    CORRELATION_WEIGHT_MAGNETIC: float = 0.4
    CORRELATION_WEIGHT_AUDIO: float = 0.4
    CORRELATION_WEIGHT_HUMANOID: float = 0.2

    # ========================================================================
    # EVENT STORAGE
    # ========================================================================
    EVENT_VIDEO_DURATION_BEFORE_SEC: int = 2
    EVENT_VIDEO_DURATION_AFTER_SEC: int = 3
    EVENT_VIDEO_FPS: int = 30
    EVENT_VIDEO_CODEC: str = "mp4v"

    # ========================================================================
    # MACHINE LEARNING
    # ========================================================================
    ML_BATCH_SIZE: int = 16
    ML_NUM_EPOCHS: int = 50
    ML_LEARNING_RATE: float = 0.001
    ML_WEIGHT_DECAY: float = 0.0001

    # Video
    ML_VIDEO_FRAMES: int = 150  # 5 segundos @ 30 FPS
    ML_VIDEO_SIZE: int = 224

    # Audio
    ML_AUDIO_DURATION_SEC: int = 5
    ML_AUDIO_N_MELS: int = 128
    ML_AUDIO_EMBEDDING_DIM: int = 256

    # Sensor
    ML_SENSOR_FEATURES: int = 15
    ML_SENSOR_EMBEDDING_DIM: int = 64

    # Classes
    ML_NUM_CLASSES: int = 4
    ML_CLASS_NAMES: list[str] = [
        "Ruído_Ambiente",
        "Interferência_Eletrônica",
        "Anomalia_Correlacionada",
        "Forma_Humanoide_Potencial"
    ]

    # ========================================================================
    # WEIGHTS & BIASES
    # ========================================================================
    WANDB_API_KEY: Optional[str] = None
    WANDB_PROJECT: str = "spectral"
    WANDB_ENTITY: Optional[str] = None
    WANDB_ENABLED: bool = False

    # ========================================================================
    # SECURITY
    # ========================================================================
    JWT_SECRET_KEY: str = "change_this_secret_key_in_production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24

    # Rate limiting
    RATE_LIMIT_WEBSOCKET_PACKETS_PER_SEC: int = 10
    RATE_LIMIT_REST_REQUESTS_PER_MIN: int = 100

    # ========================================================================
    # LOGGING
    # ========================================================================
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "spectral.log"
    LOG_ROTATION: str = "10 MB"
    LOG_RETENTION: str = "7 days"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Instância global das configurações
settings = Settings()


# ============================================================================
# VALIDAÇÃO E SETUP
# ============================================================================

def validate_settings():
    """Valida configurações e cria diretórios necessários"""

    # Criar diretórios se não existirem
    settings.EVENT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODEL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    settings.LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("✅ Diretórios criados/validados")

    # Validar configurações críticas
    if settings.INFLUXDB_TOKEN == "":
        print("⚠️  INFLUXDB_TOKEN não configurado")

    if settings.JWT_SECRET_KEY == "change_this_secret_key_in_production":
        print("⚠️  JWT_SECRET_KEY usando valor padrão - ALTERAR EM PRODUÇÃO!")

    if settings.WANDB_ENABLED and not settings.WANDB_API_KEY:
        print("⚠️  WANDB_ENABLED=True mas WANDB_API_KEY não configurado")

    print(f"✅ Configurações validadas")
    print(f"   - Servidor: {settings.SERVER_HOST}:{settings.SERVER_PORT}")
    print(f"   - Debug: {settings.DEBUG}")
    print(f"   - InfluxDB: {settings.INFLUXDB_URL}")
    print(f"   - PostgreSQL: {settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}")
    print(f"   - Event Storage: {settings.EVENT_STORAGE_DIR}")


if __name__ == "__main__":
    validate_settings()
