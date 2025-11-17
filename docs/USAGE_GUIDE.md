# 📖 Guia de Uso - Spectral

## 🚀 Quick Start

### 1. Setup Inicial

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/Spectral.git
cd Spectral

# Execute o setup automatizado
chmod +x scripts/*.sh
./scripts/setup.sh
```

O script de setup irá:
- ✅ Verificar Python 3.11+
- ✅ Criar ambiente virtual
- ✅ Instalar dependências
- ✅ Configurar .env
- ✅ Criar estrutura de diretórios
- ✅ (Opcional) Iniciar Docker containers

### 2. Configuração

Edite `server/.env` com suas configurações:

```bash
cd server
nano .env  # ou seu editor preferido
```

**Configurações importantes**:

```env
# Servidor
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
DEBUG=True

# InfluxDB (opcional)
INFLUXDB_URL=http://localhost:8086
INFLUXDB_TOKEN=seu_token_aqui

# PostgreSQL (opcional)
POSTGRES_URL=postgresql://spectral:senha@localhost:5432/spectral
```

### 3. Iniciar Servidor

**Opção 1: Script automatizado (recomendado)**

```bash
./scripts/start_server.sh
```

**Opção 2: Manual**

```bash
cd server
source venv/bin/activate
python main.py
```

O servidor estará disponível em:
- 🌐 API: http://localhost:8000
- 📚 Documentação: http://localhost:8000/docs
- 📊 Estatísticas: http://localhost:8000/stats

---

## 📱 Cliente Android

### Calibração de Sensores

**Antes de usar o app, calibre os sensores!**

1. Abra o app Spectral
2. Vá em **Settings** → **Calibrate Sensors**
3. Siga o wizard step-by-step:

#### Magnetômetro (6 passos + figura 8)
1. **Face para cima**: Coloque o celular com tela para cima
2. **Face para baixo**: Vire o celular
3. **Lado esquerdo**: Apoie no lado esquerdo
4. **Lado direito**: Apoie no lado direito
5. **Topo**: Apoie no topo (onde fica a câmera)
6. **Fundo**: Apoie no fundo (porta USB)
7. **Figura 8**: Faça movimentos de ∞ no ar

#### Acelerômetro
- Coloque em superfície **completamente plana**
- NÃO toque no celular durante calibração

#### Giroscópio
- Mantenha celular **completamente imóvel**
- NÃO toque durante calibração

### Conectar ao Servidor

1. Vá em **Settings** → **Server Configuration**
2. Digite o IP do servidor (ex: `192.168.1.100`)
3. Porta padrão: `8000`
4. Clique em **Connect**
5. Aguarde confirmação: ✅ "Connected"

---

## 🧪 Testes

### Testar Detecção de Anomalia

**Via REST API:**

```bash
curl -X POST http://localhost:8000/test/anomaly \
  -H "Content-Type: application/json" \
  -d '{
    "magnitude": 95.5,
    "audio_peak": 0.9,
    "humanoid_detected": false
  }'
```

**Resposta esperada:**

```json
{
  "input": {
    "raw_magnitude": 95.5,
    "audio_peak": 0.9,
    "humanoid_detected": false
  },
  "processing": {
    "filtered_magnitude": 94.8,
    "mean": 50.2,
    "std": 5.1,
    "threshold": 65.5
  },
  "result": {
    "anomaly_detected": true,
    "z_score": 8.7
  }
}
```

### Testar WebSocket (Python)

```python
import asyncio
import json
from websockets import connect

async def test_websocket():
    async with connect('ws://localhost:8000/ws/test_client') as ws:
        # Enviar pacote de teste
        packet = {
            "timestamp": 1705501234567890000,
            "device_id": "test_client",
            "magnetometer": {
                "x": 20.5,
                "y": -15.3,
                "z": 45.2,
                "magnitude": 75.5
            },
            "audio_peak": 0.85,
            "humanoid_detected": False
        }

        await ws.send(json.dumps(packet))

        # Receber resposta
        response = await ws.recv()
        print(f"Resposta: {response}")

asyncio.run(test_websocket())
```

---

## 🔧 Algoritmos Avançados

### Filtro de Kalman

O servidor usa **Filtro de Kalman Adaptativo** para suavizar dados do magnetômetro:

```python
from server.core.kalman_filter import AdaptiveKalmanFilter

kalman = AdaptiveKalmanFilter(
    process_variance=1e-5,
    initial_measurement_variance=1e-2,
    adaptation_rate=0.1
)

# Processar medição
filtered_value = kalman.process(raw_magnitude)
```

**Vantagens**:
- Remove ruído do sensor
- Suaviza flutuações
- Adapta-se automaticamente ao nível de ruído

### Análise de Áudio - Múltiplas Variantes

O servidor usa **6 variantes** de análise de áudio simultaneamente:

1. **FFT Clássica**: Detecção de infrassom (<20Hz) e ultrassom (>18kHz)
2. **STFT**: Análise tempo-frequência, detecta transientes
3. **Wavelet**: Análise multi-resolução em diferentes escalas
4. **Formantes (EVP)**: Detecta estrutura de fala usando LPC
5. **Filterbank**: Energia em sub-bandas (Mel-spectrogram)
6. **Zero Crossing**: Análise de periodicidade

**Ensemble (Combinação)**:

```python
from server.processing.audio_variants import AudioEnsemble

ensemble = AudioEnsemble(sample_rate=44100)
result = ensemble.analyze(audio_array)

print(result['anomaly_detected'])
print(result['confidence'])
print(result['individual_results'])
```

**Output**:
```json
{
  "anomaly_detected": true,
  "confidence": 0.78,
  "num_variants": 6,
  "num_anomalies_detected": 5,
  "individual_results": [
    {"variant": "fft_classic", "anomaly": true, "confidence": 0.8},
    {"variant": "stft_temporal", "anomaly": true, "confidence": 0.6},
    {"variant": "formant_evp", "anomaly": true, "confidence": 0.9},
    ...
  ]
}
```

---

## 📊 Monitoramento

### Estatísticas do Servidor

```bash
curl http://localhost:8000/stats
```

```json
{
  "stats": {
    "total_clients": 3,
    "total_packets_received": 1547,
    "total_events_detected": 12,
    "clients_online": 2
  },
  "clients": ["OPPO_Reno_11_01", "device_02"]
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "clients_online": 2,
  "total_packets": 1547,
  "total_events": 12
}
```

---

## 🐞 Troubleshooting

### Erro: "ModuleNotFoundError: No module named 'fastapi'"

**Solução**: Instalar dependências

```bash
cd server
source venv/bin/activate
pip install -r requirements.txt
```

### Erro: "Address already in use"

**Solução**: Porta 8000 está ocupada

```bash
# Opção 1: Encontrar e matar processo
lsof -ti:8000 | xargs kill -9

# Opção 2: Mudar porta no .env
SERVER_PORT=8001
```

### Cliente não conecta ao servidor

**Checklist**:

1. ✅ Servidor está rodando?
   ```bash
   curl http://localhost:8000/health
   ```

2. ✅ Firewall permite conexões na porta 8000?
   ```bash
   sudo ufw allow 8000
   ```

3. ✅ Cliente e servidor estão na mesma rede?
   ```bash
   # No servidor, descobrir IP
   ip addr show
   ```

4. ✅ IP correto no app Android?
   - Use IP local (ex: 192.168.1.100)
   - NÃO use localhost ou 127.0.0.1

### Calibração do magnetômetro não funciona

**Dicas**:

1. ⚠️ **Afaste-se de objetos metálicos**:
   - Mesas de metal
   - Notebooks
   - Relógios
   - Fones magnéticos

2. ⚠️ **Evite interferências**:
   - Alto-falantes
   - Ímãs
   - Motores elétricos

3. ⚠️ **Ambiente ideal**:
   - Mesa de madeira ou plástico
   - Longe de eletrônicos
   - Sem objetos metálicos próximos

---

## 📚 Recursos Adicionais

### Documentação Técnica

- [ARCHITECTURE.md](ARCHITECTURE.md) - Arquitetura completa
- [CLIENT_SPEC.md](CLIENT_SPEC.md) - Cliente Android
- [SERVER_SPEC.md](SERVER_SPEC.md) - Servidor Python
- [API_PROTOCOL.md](API_PROTOCOL.md) - Protocolos de comunicação
- [AI_ML_SPEC.md](AI_ML_SPEC.md) - Pipeline de Machine Learning

### Exemplos de Código

- `server/core/kalman_filter.py` - Implementação de Kalman
- `server/processing/audio_variants.py` - Variantes de áudio
- `client/android/CalibrationManager.kt` - Calibração Android

### Logs

```bash
# Ver logs em tempo real
tail -f logs/spectral.log

# Buscar erros
grep ERROR logs/spectral.log

# Buscar eventos
grep "ANOMALIA DETECTADA" logs/spectral.log
```

---

## 🆘 Suporte

- 📧 Email: contact@spectral-project.dev
- 🐛 Issues: https://github.com/spectral-project/issues
- 📖 Wiki: https://github.com/spectral-project/wiki

---

**Última Atualização**: 2025-01-17
**Versão**: 1.0.0
