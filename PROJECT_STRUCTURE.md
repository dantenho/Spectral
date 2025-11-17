# 📂 Estrutura do Projeto Spectral

```
Spectral/
├── 📄 README.md                    # Documentação principal
├── 📄 LICENSE                      # Licença MIT
├── 📄 CONTRIBUTING.md              # Guia de contribuição
├── 📄 PROJECT_STRUCTURE.md         # Este arquivo
├── 📄 .gitignore                   # Arquivos ignorados pelo Git
├── 📄 docker-compose.yml           # Orquestração de containers
│
├── 📁 docs/                        # Documentação técnica
│   ├── ARCHITECTURE.md             # Arquitetura geral do sistema
│   ├── CLIENT_SPEC.md              # Especificação do cliente Android
│   ├── SERVER_SPEC.md              # Especificação do servidor
│   ├── API_PROTOCOL.md             # Protocolos de comunicação
│   └── AI_ML_SPEC.md               # Pipeline de Machine Learning
│
├── 📁 client/                      # Cliente Android
│   └── android/
│       ├── app/
│       │   ├── src/
│       │   │   ├── main/
│       │   │   │   ├── java/com/spectral/
│       │   │   │   │   ├── data/              # Camada de dados
│       │   │   │   │   │   ├── model/
│       │   │   │   │   │   ├── repository/
│       │   │   │   │   │   └── remote/
│       │   │   │   │   ├── domain/            # Lógica de negócio
│       │   │   │   │   │   ├── usecase/
│       │   │   │   │   │   └── mapper/
│       │   │   │   │   ├── presentation/      # UI
│       │   │   │   │   │   ├── ui/
│       │   │   │   │   │   └── components/
│       │   │   │   │   └── utils/             # Utilidades
│       │   │   │   ├── res/                   # Recursos Android
│       │   │   │   └── AndroidManifest.xml
│       │   │   └── test/                      # Testes
│       │   └── build.gradle.kts
│       ├── gradle/
│       ├── build.gradle.kts
│       └── settings.gradle.kts
│
├── 📁 server/                      # Servidor Backend (Python)
│   ├── main.py                     # Entry point
│   ├── requirements.txt            # Dependências Python
│   ├── .env.example                # Variáveis de ambiente exemplo
│   ├── Dockerfile                  # Container do servidor
│   │
│   ├── config/                     # Configurações
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── logging.py
│   │
│   ├── api/                        # Camada de API
│   │   ├── __init__.py
│   │   ├── websocket.py            # WebSocket handlers
│   │   ├── rest.py                 # REST endpoints
│   │   └── schemas.py              # Pydantic models
│   │
│   ├── core/                       # Lógica principal
│   │   ├── __init__.py
│   │   ├── anomaly_detection.py   # Engine de detecção
│   │   ├── event_manager.py        # Gerenciador de eventos
│   │   ├── buffer_manager.py       # Buffers circulares
│   │   └── correlation_engine.py   # Correlação multi-sensorial
│   │
│   ├── processing/                 # Processamento de dados
│   │   ├── __init__.py
│   │   ├── audio_processor.py     # FFT, EVP
│   │   ├── video_processor.py     # Extração de frames
│   │   ├── magnetic_processor.py  # Análise magnética
│   │   └── sensor_fusion.py       # Fusão de dados
│   │
│   ├── database/                   # Camada de dados
│   │   ├── __init__.py
│   │   ├── influxdb_client.py     # Time series
│   │   ├── postgres_client.py     # Relacional
│   │   └── models.py              # SQLAlchemy models
│   │
│   ├── ml/                         # Machine Learning
│   │   ├── __init__.py
│   │   ├── dataset.py             # PyTorch Dataset
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── video_encoder.py   # EfficientNet
│   │   │   ├── audio_encoder.py   # 1D CNN
│   │   │   └── fusion_classifier.py # MLP/LSTM
│   │   ├── training.py            # Lightning Module
│   │   └── inference.py           # Produção
│   │
│   ├── storage/                    # Armazenamento
│   │   ├── __init__.py
│   │   ├── event_storage.py       # Sistema de arquivos
│   │   └── video_buffer.py        # Buffer de vídeo
│   │
│   └── tests/                      # Testes
│       ├── __init__.py
│       ├── test_api.py
│       ├── test_anomaly.py
│       └── test_ml.py
│
├── 📁 interface/                   # Interface Gradio
│   ├── gradio_app.py               # Aplicação principal
│   ├── components/                 # Componentes UI
│   │   ├── ar_mode.py
│   │   ├── field_vector.py
│   │   ├── audio_analysis.py
│   │   ├── timeline.py
│   │   └── ai_lab.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── 📁 models/                      # Modelos treinados
│   ├── checkpoints/                # Checkpoints de treinamento
│   │   └── .gitkeep
│   ├── pretrained/                 # Modelos pré-treinados
│   └── production/                 # Modelos em produção
│
├── 📁 data/                        # Dados
│   ├── events/                     # Eventos detectados
│   │   ├── .gitkeep
│   │   └── event_YYYYMMDD_HHMMSS/
│   │       ├── video.mp4
│   │       ├── audio.wav
│   │       ├── sensors.csv
│   │       └── metadata.json
│   ├── training/                   # Dados de treinamento
│   │   ├── .gitkeep
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── raw/                        # Dados brutos
│
├── 📁 scripts/                     # Scripts utilitários
│   ├── setup_env.sh                # Setup de ambiente
│   ├── start_server.sh             # Iniciar servidor
│   ├── train_model.py              # Treinar modelo
│   └── export_model.py             # Exportar modelo
│
├── 📁 configs/                     # Arquivos de configuração
│   ├── train_config.yaml           # Config de treinamento
│   ├── inference_config.yaml       # Config de inferência
│   └── deployment_config.yaml      # Config de deploy
│
└── 📁 logs/                        # Logs
    └── .gitkeep

```

## 📊 Descrição dos Módulos

### Cliente Android
Aplicativo móvel responsável por:
- Coleta de dados dos sensores
- Processamento Edge AI (NPU)
- Transmissão para servidor
- Interface do usuário

### Servidor Backend
Backend Python que executa:
- Recepção de dados via WebSocket
- Detecção de anomalias
- Armazenamento de eventos
- Pipeline de ML

### Interface Gradio
Dashboard web para:
- Visualização em tempo real
- Análise de dados
- Controle de treinamento
- Gerenciamento de eventos

### Banco de Dados
- **InfluxDB**: Dados de time series dos sensores
- **PostgreSQL**: Metadados de eventos e classificações

### Machine Learning
Pipeline completo de:
- Dataset preparation
- Treinamento de modelos
- Avaliação e métricas
- Deploy em produção

## 🔧 Tecnologias por Módulo

| Módulo | Tecnologias Principais |
|--------|------------------------|
| Cliente | Kotlin, Ktor, TensorFlow Lite, CameraX |
| Servidor | Python, FastAPI, PyTorch, Uvicorn |
| Interface | Gradio, Plotly, Matplotlib |
| Banco de Dados | InfluxDB, PostgreSQL, SQLAlchemy |
| ML | PyTorch, Lightning, Weights & Biases |
| DevOps | Docker, Docker Compose, Git |

## 📈 Fluxo de Dados

```
Android App → WebSocket → Backend Server → Database
                              ↓
                         Event Storage
                              ↓
                         ML Training
                              ↓
                         Trained Model
                              ↓
                         Inference API
```

## 🚀 Quick Start

```bash
# 1. Setup servidor
cd server
pip install -r requirements.txt
python main.py

# 2. Setup interface
cd interface
pip install -r requirements.txt
python gradio_app.py

# 3. Build Android app
cd client/android
./gradlew assembleDebug
```

---

**Última Atualização**: 2025-01-17
**Versão**: 1.0.0
