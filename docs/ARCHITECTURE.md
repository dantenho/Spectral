# 🏗️ Arquitetura do Sistema Spectral

## 📋 Visão Geral

**Spectral** é um sistema de detecção de anomalias ambientais em tempo real que combina múltiplos sensores, processamento de edge AI e análise avançada de machine learning para identificar e catalogar eventos anômalos.

## 🎯 Objetivos do Sistema

1. **Coleta Multi-Sensorial**: Capturar dados sincronizados de magnetômetro, áudio, vídeo, acelerômetro, giroscópio, Bluetooth e NFC
2. **Detecção em Tempo Real**: Identificar anomalias correlacionadas entre diferentes sensores
3. **Edge AI**: Processar dados localmente no dispositivo usando NPU para pré-filtros inteligentes
4. **Armazenamento de Eventos**: Catalogar eventos anômalos com metadados ricos
5. **Aprendizado Contínuo**: Treinar modelos de IA para reconhecer padrões em eventos detectados

---

## 🏛️ Arquitetura Geral

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENTE ANDROID                          │
│                      (OPPO Reno 11 F5)                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │Magnetô-  │  │Microfone │  │ Câmera   │  │Acel/Giro │       │
│  │metro     │  │(44.1kHz) │  │ (2K)     │  │          │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
│       │             │              │             │              │
│       └─────────────┴──────────────┴─────────────┘              │
│                         │                                       │
│              ┌──────────▼──────────┐                           │
│              │ SensorDataCollector │                           │
│              │   (Sincronização)   │                           │
│              └──────────┬──────────┘                           │
│                         │                                       │
│              ┌──────────▼──────────┐                           │
│              │  Edge AI (NPU)      │                           │
│              │  - Pose Estimation  │                           │
│              │  - TFLite/PyTorch   │                           │
│              └──────────┬──────────┘                           │
│                         │                                       │
│              ┌──────────▼──────────┐                           │
│              │  Ktor WebSocket     │                           │
│              │  Client             │                           │
│              └──────────┬──────────┘                           │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          │ WebSocket (10Hz, JSON)
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    SERVIDOR BACKEND                             │
│                   (Python + RTX 4090)                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              FastAPI + Ktor Server                       │  │
│  │              WebSocket Handler                           │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                         │
│  ┌────────────────────▼─────────────────────────────────────┐  │
│  │          Anomaly Detection Engine                        │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │  Magnetic    │  │  Audio FFT   │  │  Correlation │  │  │
│  │  │  Analysis    │  │  Analysis    │  │  Engine      │  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                         │
│                       ├─► Event Trigger Detected                │
│                       │                                         │
│  ┌────────────────────▼─────────────────────────────────────┐  │
│  │          Event Storage System                            │  │
│  │  - Video Clip (2s before, 3s after)                      │  │
│  │  - Audio RAW                                              │  │
│  │  - Sensor Data CSV                                        │  │
│  │  - Metadata JSON                                          │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                         │
│  ┌────────────────────▼─────────────────────────────────────┐  │
│  │               Database Layer                             │  │
│  │  - InfluxDB (Time Series - Sensor Data)                 │  │
│  │  - PostgreSQL (Events Metadata)                          │  │
│  └────────────────────┬─────────────────────────────────────┘  │
│                       │                                         │
│  ┌────────────────────▼─────────────────────────────────────┐  │
│  │          AI/ML Training Pipeline (PyTorch)               │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │  │
│  │  │Video Encoder │  │Audio Encoder │  │  Fusion &    │  │  │
│  │  │(EfficientNet)│  │(1D CNN)      │  │Classification│  │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          │ HTTP/WebSocket
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                  INTERFACE GRADIO                               │
│                  (Web Dashboard)                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ AR Mode  │  │  Field   │  │  Audio   │  │ Timeline │       │
│  │ (Stream) │  │ Vector   │  │ Analysis │  │ Events   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │             AI Lab (Training Control)                    │  │
│  │  - Dataset Selection                                     │  │
│  │  - Training Controls                                     │  │
│  │  - Metrics Visualization (W&B Integration)              │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Fluxo de Dados Detalhado

### Fase 1: Coleta e Transmissão (Cliente → Servidor)

```
1. [Sensores] → Dados brutos coletados a 100Hz
2. [SensorDataCollector] → Sincronização por timestamp unificado
3. [Edge Processor] → Pré-processamento local (NPU)
4. [WebSocket Client] → Empacotamento JSON (10Hz)
5. [Network] → Transmissão para servidor
```

### Fase 2: Detecção de Anomalias (Servidor)

```
1. [WebSocket Handler] → Recepção e deserialização
2. [Buffer Manager] → Janela deslizante de 5 segundos
3. [Anomaly Engine] → Análise em tempo real
   ├─ Magnetic Analysis: magnitude > média + 3σ
   ├─ Audio FFT: Picos anômalos (< 20Hz, > 18kHz)
   ├─ Correlation: Eventos simultâneos em múltiplos sensores
   └─ Humanoid Flag: Prioridade alta
4. [Trigger Decision] → Anomalia detectada?
   ├─ YES → Fase 3
   └─ NO → Continuar monitoramento
```

### Fase 3: Armazenamento de Evento (Servidor)

```
1. [Event Packager] → Coleta dados da janela de tempo
   ├─ Video: 2s antes + 3s depois (5s total)
   ├─ Audio: RAW do mesmo intervalo
   ├─ Sensors: CSV com todos os dados sincronizados
   └─ Metadata: JSON com contexto
2. [File System] → Salvar em disco estruturado
3. [Database] → Registrar metadados
   ├─ InfluxDB: Time series dos sensores
   └─ PostgreSQL: Evento e classificação
```

### Fase 4: Treinamento de IA (Servidor)

```
1. [Data Loader] → Carregar eventos salvos
2. [Video Encoder] → Extrair embeddings (EfficientNet)
3. [Audio Encoder] → Extrair embeddings (1D CNN)
4. [Sensor Processor] → Normalizar dados tabulares
5. [Fusion Model] → Concatenar embeddings
6. [Classifier] → MLP ou LSTM
7. [Training Loop] → PyTorch + RTX 4090
8. [Evaluation] → Métricas e validação
9. [Model Export] → Salvar checkpoint
```

---

## 🧩 Componentes Principais

### 1. Cliente Android (Kotlin + MVVM)

#### 1.1 SensorDataCollector
- **Responsabilidade**: Sincronizar todos os sensores com timestamp nano
- **Tecnologia**: Android SensorManager + Kotlin Coroutines
- **Taxa**: 100Hz (10ms)

#### 1.2 Edge AI Processor
- **Responsabilidade**: Executar pose estimation na NPU
- **Tecnologia**: TensorFlow Lite (MoveNet) ou PyTorch Mobile (BlazePose)
- **Output**: Boolean flag + keypoints coordinates

#### 1.3 Network Client
- **Responsabilidade**: Transmitir dados via WebSocket
- **Tecnologia**: Ktor Client
- **Formato**: JSON @ 10Hz (100ms)

### 2. Servidor Backend (Python + FastAPI)

#### 2.1 WebSocket Handler
- **Responsabilidade**: Receber e processar streams
- **Tecnologia**: FastAPI WebSocket + asyncio
- **Concorrência**: Múltiplos clientes simultâneos

#### 2.2 Anomaly Detection Engine
- **Responsabilidade**: Detectar correlações anômalas
- **Tecnologia**: NumPy, SciPy (FFT), custom algorithms
- **Critérios**:
  - Magnético: `magnitude > mean + 3*std`
  - Áudio: Picos em `< 20Hz` ou `> 18kHz`
  - Correlação: Janela de 1 segundo

#### 2.3 Event Storage System
- **Responsabilidade**: Salvar eventos em formato estruturado
- **Estrutura**:
```
data/events/
├── event_20250117_143052/
│   ├── video.mp4          # 5 segundos (2+3)
│   ├── audio.wav          # RAW 44.1kHz
│   ├── sensors.csv        # Todos os dados
│   └── metadata.json      # Contexto
```

#### 2.4 Database Layer
- **InfluxDB**: Time series de sensores (alta performance)
- **PostgreSQL**: Metadados de eventos e classificações

#### 2.5 AI Training Pipeline
- **Framework**: PyTorch + PyTorch Lightning
- **Hardware**: RTX 4090 (CUDA)
- **Monitoramento**: Weights & Biases

### 3. Interface Gradio (Python)

#### 3.1 Componentes UI
- **AR Mode**: `gr.Video()` com stream ao vivo
- **Campo Vetorial**: `gr.Plot()` com Plotly 3D
- **Análise Sonora**: Espectrograma + Osciloscópio
- **Timeline**: `gr.DataFrame()` + video player
- **AI Lab**: Controles de treinamento + gráficos W&B

---

## 🔐 Segurança e Performance

### Segurança
- **Autenticação**: JWT tokens para cliente-servidor
- **Criptografia**: TLS 1.3 para WebSocket
- **Rate Limiting**: 10 pacotes/segundo máximo
- **Validação**: Schema JSON com Pydantic

### Performance
- **Latência**: < 50ms (cliente → servidor)
- **Throughput**: Suporta 10 clientes simultâneos
- **Storage**: Compressão H.265 para vídeo
- **Database**: Índices otimizados para queries temporais

---

## 📊 Requisitos de Hardware

### Cliente (OPPO Reno 11 F5)
- **CPU**: MediaTek Dimensity 7050 (8 cores)
- **NPU**: APU 3.0 (para Edge AI)
- **RAM**: 8GB+
- **Storage**: 128GB+ (armazenamento local temporário)

### Servidor
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **CPU**: 16+ cores (para processamento paralelo)
- **RAM**: 64GB+ (buffer de eventos)
- **Storage**: 2TB+ SSD NVMe (eventos e modelos)

---

## 🚀 Roadmap de Implementação

### Fase 1: MVP (Mínimo Produto Viável) - 4-6 semanas
- ✅ Cliente Android: Coleta de sensores básicos
- ✅ Servidor: Recepção de dados via WebSocket
- ✅ Detecção de anomalia magnética simples
- ✅ Armazenamento básico de eventos
- ✅ Interface Gradio: Visualização em tempo real

### Fase 2: Edge AI + Análise Avançada - 3-4 semanas
- ⏳ Integração TFLite no cliente
- ⏳ Pose estimation em tempo real
- ⏳ Análise FFT de áudio no servidor
- ⏳ Correlação multi-sensorial
- ⏳ Database layer (InfluxDB + PostgreSQL)

### Fase 3: Machine Learning Pipeline - 4-6 semanas
- 🔜 Dataset preparation
- 🔜 Modelo de fusão multimodal
- 🔜 Training loop com PyTorch
- 🔜 Integração com Weights & Biases
- 🔜 Deploy de modelo treinado

### Fase 4: Recursos Avançados - Futuro
- 🔮 Sonar acústico (Doppler effect)
- 🔮 Análise EVP (Electronic Voice Phenomena)
- 🔮 Multi-client collaboration
- 🔮 Cloud deployment (AWS/GCP)

---

## 📚 Referências Técnicas

### Sensores e Fusão
- [Android Sensor Fusion](https://developer.android.com/guide/topics/sensors/sensors_position)
- [Madgwick Filter](https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/)

### Edge AI
- [TensorFlow Lite Guide](https://www.tensorflow.org/lite)
- [MoveNet: Ultra fast and accurate pose detection](https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html)

### Audio Processing
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [EVP Analysis Guide (ASSAP)](https://www.assap.ac.uk/articles/detail/analysing-evp-and-paranormal-sound-recordings)

### Machine Learning
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Multimodal Fusion Techniques](https://arxiv.org/abs/2103.05561)

---

**Última Atualização**: 2025-01-17
**Versão**: 1.0.0
**Arquiteto**: Sistema Spectral Team
