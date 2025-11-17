# 👻 Spectral - Sistema de Detecção de Anomalias Ambientais

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Kotlin](https://img.shields.io/badge/kotlin-1.9+-purple.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red.svg)

**Sistema avançado de detecção de anomalias em tempo real utilizando fusão multi-sensorial, Edge AI e Deep Learning**

[Documentação](#-documentação) •
[Arquitetura](#-arquitetura) •
[Instalação](#-instalação) •
[Uso](#-uso) •
[Roadmap](#-roadmap)

</div>

---

## 🎯 Visão Geral

**Spectral** é um sistema completo de detecção de anomalias ambientais que combina:

- 📱 **Cliente Android** (OPPO Reno 11 F5) com coleta multi-sensorial em tempo real
- 🖥️ **Servidor Backend** (Python + RTX 4090) para processamento e análise
- 🧠 **Pipeline de IA** para classificação automática de eventos
- 🎨 **Interface Gradio** para visualização e controle

### Características Principais

- ✅ **Coleta Multi-Sensorial**: Magnetômetro, áudio, vídeo, acelerômetro, giroscópio, Bluetooth, NFC
- ✅ **Edge AI**: Pose estimation em tempo real usando NPU do dispositivo
- ✅ **Detecção Correlacionada**: Análise multi-sensorial com janelas temporais
- ✅ **Armazenamento Estruturado**: Eventos salvos com vídeo, áudio e metadados
- ✅ **Deep Learning**: Modelo de fusão multimodal (vídeo + áudio + sensores)
- ✅ **Tempo Real**: Latência < 50ms, streaming a 10Hz

---

## 🏗️ Arquitetura

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│  Cliente Android│         │  Servidor       │         │  Interface      │
│  (Kotlin/MVVM)  │◀───────▶│  (FastAPI)      │◀───────▶│  (Gradio)       │
│                 │         │                 │         │                 │
│  • Sensores     │  WebSocket  • Anomaly      │   HTTP  │  • AR Mode      │
│  • Edge AI (NPU)│  10Hz   │   Detection     │         │  • Analytics    │
│  • Streaming    │         │  • Event Storage│         │  • AI Lab       │
└─────────────────┘         │  • ML Pipeline  │         └─────────────────┘
                            └─────────────────┘
```

### Componentes

1. **Cliente Android** (Kotlin)
   - Coleta de dados de múltiplos sensores @ 100Hz
   - Pose estimation em tempo real (TensorFlow Lite)
   - Transmissão via WebSocket @ 10Hz
   - Streaming de vídeo 2K (30 FPS)

2. **Servidor Backend** (Python)
   - Recepção e processamento assíncrono (FastAPI)
   - Detecção de anomalias magnéticas e sonoras
   - Correlação multi-sensorial
   - Armazenamento de eventos (InfluxDB + PostgreSQL)
   - Pipeline de treinamento de IA (PyTorch + RTX 4090)

3. **Interface Gradio** (Python)
   - Visualização em tempo real
   - Análise espectral de áudio
   - Campo vetorial magnético 3D
   - Timeline de eventos
   - Controles de treinamento de IA

---

## 📚 Documentação

Documentação técnica completa disponível em `/docs`:

| Documento | Descrição |
|-----------|-----------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Arquitetura geral do sistema |
| [CLIENT_SPEC.md](docs/CLIENT_SPEC.md) | Especificação do cliente Android |
| [SERVER_SPEC.md](docs/SERVER_SPEC.md) | Especificação do servidor backend |
| [API_PROTOCOL.md](docs/API_PROTOCOL.md) | Protocolos de comunicação |
| [AI_ML_SPEC.md](docs/AI_ML_SPEC.md) | Pipeline de Machine Learning |

---

## 🚀 Instalação

### Pré-requisitos

#### Hardware
- **Cliente**: OPPO Reno 11 F5 (ou similar com NPU)
- **Servidor**:
  - GPU: NVIDIA RTX 4090 (ou RTX 3090/4080)
  - CPU: 16+ cores
  - RAM: 64GB+
  - Storage: 2TB+ SSD NVMe

#### Software
- **Cliente**: Android 8.0+ (API 26+)
- **Servidor**:
  - Python 3.11+
  - CUDA 12.1+
  - Docker (opcional)

### Setup do Servidor

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/Spectral.git
cd Spectral/server

# Criar ambiente virtual
python3.11 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis de ambiente
cp .env.example .env
# Edite .env com suas configurações

# Iniciar servidor
python main.py
```

### Setup do Cliente Android

```bash
cd client/android

# Build com Gradle
./gradlew assembleDebug

# Instalar no dispositivo
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### Docker (Opcional)

```bash
# Servidor
docker-compose up -d

# Verificar logs
docker-compose logs -f
```

---

## 📊 Uso

### 1. Iniciar Sistema

```bash
# Terminal 1: Servidor Backend
cd server
python main.py

# Terminal 2: Interface Gradio
cd interface
python gradio_app.py
```

### 2. Conectar Cliente Android

1. Abra o app Spectral no dispositivo
2. Configure IP do servidor (Settings)
3. Clique em "Connect"
4. Aguarde confirmação de conexão

### 3. Monitorar Dados

Acesse a interface Gradio em:
```
http://localhost:7860
```

**Abas Disponíveis**:
- **AR Mode**: Stream de vídeo em tempo real
- **Campo Vetorial**: Visualização 3D do magnetômetro
- **Análise Sonora**: Espectrograma e osciloscópio
- **Timeline**: Histórico de eventos detectados
- **AI Lab**: Controles de treinamento

### 4. Treinar Modelo de IA

```bash
cd server/ml
python training.py --config configs/train_config.yaml
```

Ou via interface Gradio:
1. Acesse aba "AI Lab"
2. Selecione dataset de treinamento
3. Ajuste hiperparâmetros
4. Clique em "Start Training"

---

## 🧪 Exemplos

### Enviar Pacote de Dados (Python)

```python
import asyncio
import json
from websockets import connect

async def send_sensor_data():
    async with connect('ws://localhost:8000/ws/device_01') as websocket:
        packet = {
            "timestamp": 1705501234567890000,
            "device_id": "device_01",
            "magnetometer": {
                "x": 0.123, "y": -0.456, "z": 0.789,
                "magnitude": 0.936
            },
            "audio_peak": 0.75,
            "humanoid_detected": False
        }
        await websocket.send(json.dumps(packet))
        response = await websocket.recv()
        print(f"Server: {response}")

asyncio.run(send_sensor_data())
```

### Query de Eventos (REST API)

```bash
# Listar eventos de hoje
curl -X GET "http://localhost:8000/api/v1/events?start_date=2025-01-17"

# Obter evento específico
curl -X GET "http://localhost:8000/api/v1/events/event_20250117_143052"

# Download de vídeo do evento
curl -X GET "http://localhost:8000/api/v1/events/event_20250117_143052/video" \
     --output event_video.mp4
```

---

## 🔬 Sensores Suportados

| Sensor | Frequência | Precisão | Uso |
|--------|------------|----------|-----|
| **Magnetômetro** | 100 Hz | ±0.1 µT | Detecção de anomalias magnéticas |
| **Microfone** | 44.1 kHz | 16-bit | Análise espectral (FFT, EVP) |
| **Câmera** | 30 FPS | 1920x1080 | Pose estimation, evidência visual |
| **Acelerômetro** | 100 Hz | ±0.01 m/s² | Fusão de sensores |
| **Giroscópio** | 100 Hz | ±0.001 rad/s | Orientação da câmera |
| **Bluetooth** | 1 Hz | - | Detecção de dispositivos |
| **NFC** | On-demand | - | Tags NFC |

---

## 🧠 Modelo de IA

### Arquitetura

**Fusão Multimodal** (Vídeo + Áudio + Sensores)

```
Video (EfficientNet-B0) ─┐
                          ├─► Fusion MLP ─► Classifier
Audio (1D CNN) ──────────┤    [1600→128]    [4 classes]
                          │
Sensors (MLP) ────────────┘
```

### Classes de Classificação

1. **Ruído_Ambiente**: Ruído natural sem correlação
2. **Interferência_Eletrônica**: Dispositivos eletrônicos
3. **Anomalia_Correlacionada**: Evento de interesse
4. **Forma_Humanoide_Potencial**: Detecção + anomalia

### Performance

- **Accuracy**: > 85% (target)
- **Inference Time**: < 200ms
- **Model Size**: ~100 MB

---

## 📈 Roadmap

### Fase 1: MVP ✅ (4-6 semanas)
- [x] Cliente Android com coleta básica
- [x] Servidor com WebSocket
- [x] Detecção de anomalia magnética
- [x] Interface Gradio
- [ ] Documentação completa

### Fase 2: Edge AI + Análise Avançada 🔄 (3-4 semanas)
- [ ] Integração TensorFlow Lite
- [ ] Pose estimation em tempo real
- [ ] Análise FFT de áudio
- [ ] Correlação multi-sensorial
- [ ] Database layer (InfluxDB + PostgreSQL)

### Fase 3: Machine Learning 🔜 (4-6 semanas)
- [ ] Dataset preparation
- [ ] Modelo de fusão multimodal
- [ ] Training loop
- [ ] Integração W&B
- [ ] Deploy do modelo

### Fase 4: Recursos Avançados 🔮 (Futuro)
- [ ] Sonar acústico (Doppler)
- [ ] Análise EVP avançada
- [ ] Multi-client collaboration
- [ ] Cloud deployment (AWS/GCP)

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Por favor:

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

---

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

---

## 🙏 Agradecimentos

### Frameworks e Bibliotecas
- [PyTorch](https://pytorch.org/) - Deep Learning framework
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework moderno
- [Gradio](https://gradio.app/) - Interface de ML
- [Ktor](https://ktor.io/) - Kotlin networking
- [TensorFlow Lite](https://www.tensorflow.org/lite) - Edge AI

### Pesquisa
- [ASSAP](https://www.assap.ac.uk/) - Guia de análise EVP
- [Social Voice Project](https://thesocialvoiceproject.org/) - Análise forense de voz
- [Madgwick Filter](https://x-io.co.uk/) - Fusão de sensores

---

## 📞 Contato

**Projeto Spectral** - Detecção de Anomalias Ambientais

- GitHub: [@spectral-project](https://github.com/spectral-project)
- Email: contact@spectral-project.dev

---

<div align="center">

**Desenvolvido com IA e Ciência**

*Versão 1.0.0 - Janeiro 2025*

</div>
