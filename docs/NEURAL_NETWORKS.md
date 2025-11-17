# 🧠 Sistema de Múltiplas Redes Neurais - Spectral

## 📋 Visão Geral

O Spectral implementa um **sistema modular de múltiplas redes neurais** que podem ser usadas **individualmente** ou **combinadas em ensemble** para maximizar a accuracy de classificação de eventos anômalos.

---

## 🏗️ Arquitetura Modular

### Componentes Principais

```
┌─────────────────────────────────────────────────────────────┐
│                    SISTEMA DE ML                             │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │  Video Encoder  │  │  Audio Encoder  │  │   Sensor    │ │
│  │                 │  │                 │  │   Encoder   │ │
│  │ • EfficientNet  │  │ • CNN           │  │ • MLP       │ │
│  │ • ResNet        │  │ • ResNet        │  │ • ResNet    │ │
│  │ • MobileNet     │  │ • Transformer   │  │ • Attention │ │
│  │ • ConvNeXt      │  │                 │  │             │ │
│  │ • ViT           │  │                 │  │             │ │
│  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘ │
│           │                    │                   │         │
│           └────────────────────┴───────────────────┘         │
│                              │                               │
│                   ┌──────────▼──────────┐                   │
│                   │  Fusion Classifier  │                   │
│                   │                     │                   │
│                   │ • Concat            │                   │
│                   │ • Attention         │                   │
│                   │ • Gated             │                   │
│                   │ • Bilinear          │                   │
│                   └──────────┬──────────┘                   │
│                              │                               │
│                   ┌──────────▼──────────┐                   │
│                   │    Classificação    │                   │
│                   │    (4 classes)      │                   │
│                   └─────────────────────┘                   │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                    ENSEMBLE                                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Combina múltiplos modelos completos:                       │
│                                                               │
│  • Voting (majoritária)                                     │
│  • Average (média de probabilidades)                         │
│  • Weighted (média ponderada)                               │
│  • Stacking (meta-learner)                                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Modelos Disponíveis

### 1. Video Encoder (6 variantes)

| Variante | Backbone | Params | Speed | Uso |
|----------|----------|--------|-------|-----|
| **EfficientNet-B0** | efficientnet_b0 | ~5M | ⚡⚡⚡ | Padrão, balanceado |
| **EfficientNet-B2** | efficientnet_b2 | ~9M | ⚡⚡ | Melhor accuracy |
| **ResNet50** | resnet50 | ~25M | ⚡⚡ | Robusto, clássico |
| **MobileNetV3** | mobilenetv3_large | ~5M | ⚡⚡⚡⚡ | Ultra rápido |
| **ConvNeXt** | convnext_tiny | ~28M | ⚡ | Estado da arte |
| **ViT** | vit_small_patch16 | ~22M | ⚡ | Atenção global |

**Agregação Temporal**:
- `mean`: Média simples (rápido)
- `max`: Máximo (detecta picos)
- `lstm`: LSTM 2 camadas (sequencial)
- `attention`: Multi-head attention (melhor)

### 2. Audio Encoder (4 variantes)

| Variante | Arquitetura | Embedding | Speed | Uso |
|----------|-------------|-----------|-------|-----|
| **CNN Small** | 4 Conv layers | 256 | ⚡⚡⚡ | Rápido |
| **CNN Large** | 4 Conv layers | 512 | ⚡⚡ | Melhor features |
| **ResNet** | Residual blocks | 256 | ⚡⚡ | Robusto |
| **Transformer** | Self-attention | 256 | ⚡ | Dependências temporais |

**Processamento**:
- Mel-spectrogram (n_mels: 64-128)
- FFT size: 2048
- Hop length: 512
- Log-scale (dB)

### 3. Sensor Encoder (3 variantes)

| Variante | Arquitetura | Params | Uso |
|----------|-------------|--------|-----|
| **Simple MLP** | 2 hidden layers | ~20K | Padrão, rápido |
| **Deep ResNet** | 4 residual blocks | ~80K | Dados complexos |
| **Attention** | Transformer | ~100K | Interdependências |

---

## 🔀 Fusion Strategies

### 1. Concat (Concatenação)

```python
video_emb + audio_emb + sensor_emb → MLP → classes
```

**Prós**: Simples, rápido
**Contras**: Não captura interações

### 2. Attention (Atenção)

```python
MultiHeadAttention(video, audio, sensor) → MLP → classes
```

**Prós**: Captura relações entre modalidades
**Contras**: Mais lento

### 3. Gated (Com gates)

```python
video * gate(video) + audio * gate(audio) + sensor * gate(sensor) → MLP
```

**Prós**: Aprende importância de cada modalidade
**Contras**: Pode overfittar

### 4. Bilinear

```python
video ⊗ audio + video ⊗ sensor + audio ⊗ sensor → MLP
```

**Prós**: Captura interações de 2ª ordem
**Contras**: Muitos parâmetros

---

## 🎭 Variantes Pré-Configuradas

### FusionVariants

```python
from ml.models import FusionVariants

# 1. Lightweight: Rápido e eficiente
model = FusionVariants.lightweight(num_classes=4)
# - MobileNetV3 + CNN Small + Simple MLP
# - Fusion: Concat
# - Params: ~8M
# - Speed: ~50ms

# 2. Balanced: Balanço velocidade/accuracy
model = FusionVariants.balanced(num_classes=4)
# - EfficientNet-B0 + ResNet + Deep MLP
# - Fusion: Attention
# - Params: ~12M
# - Speed: ~100ms

# 3. Accurate: Máxima accuracy
model = FusionVariants.accurate(num_classes=4)
# - EfficientNet-B2 + Transformer + Attention
# - Fusion: Bilinear
# - Params: ~25M
# - Speed: ~200ms

# 4. Vision Focused: Foco em vídeo
model = FusionVariants.vision_focused(num_classes=4)
# - ConvNeXt + CNN Small + Simple MLP
# - Fusion: Gated
# - Para detecção de forma humanoide

# 5. Audio Focused: Foco em áudio
model = FusionVariants.audio_focused(num_classes=4)
# - MobileNetV3 + CNN Large + Simple MLP
# - Fusion: Attention
# - Para análise EVP
```

---

## 🎪 Sistema de Ensemble

### Métodos de Combinação

#### 1. Voting (Votação Majoritária)

```python
# Hard voting: cada modelo vota em uma classe
# Classe mais votada vence
ensemble = NeuralEnsemble(models, method='voting')
```

**Uso**: Modelos bem diferentes, confiança binária

#### 2. Average (Média)

```python
# Soft voting: média das probabilidades
# P_ensemble = mean(P_model1, P_model2, ...)
ensemble = NeuralEnsemble(models, method='average')
```

**Uso**: Modelos similares, padrão recomendado

#### 3. Weighted (Ponderado)

```python
# Média ponderada por pesos
weights = [0.3, 0.5, 0.2]  # modelo 2 tem mais peso
ensemble = NeuralEnsemble(models, method='weighted', weights=weights)
```

**Uso**: Alguns modelos são melhores que outros

#### 4. Stacking (Meta-Learner)

```python
# MLP aprende a combinar predições
ensemble = NeuralEnsemble(models, method='stacking')
```

**Uso**: Máxima performance, precisa treinar meta-learner

---

## 📊 Ensembles Pré-Configurados

### EnsembleVariants

```python
from ml.ensemble import EnsembleVariants

# 1. Fast Ensemble (3 modelos)
ensemble = EnsembleVariants.fast_ensemble()
# - Lightweight + Balanced + Vision Focused
# - Method: Average
# - Speed: ~150ms
# - Accuracy: +3-5% vs single model

# 2. Accurate Ensemble (3 modelos)
ensemble = EnsembleVariants.accurate_ensemble()
# - Balanced + Accurate + Audio Focused
# - Method: Weighted [0.3, 0.5, 0.2]
# - Speed: ~300ms
# - Accuracy: +5-8% vs single model

# 3. Full Ensemble (5 modelos + meta-learner)
ensemble = EnsembleVariants.full_ensemble()
# - Todos os 5 modelos + Stacking
# - Method: Stacking
# - Speed: ~500ms
# - Accuracy: +8-12% vs single model

# 4. Specialized Ensemble (EVP/Humanoide)
ensemble = EnsembleVariants.specialized_ensemble()
# - Vision (20%) + Audio (50%) + Balanced (30%)
# - Method: Weighted
# - Otimizado para detecção EVP
```

---

## 💻 Uso Prático

### Treinamento de Modelo Único

```python
import torch
from ml.models import FusionVariants

# Criar modelo
model = FusionVariants.balanced(num_classes=4)

# Dados de exemplo
video = torch.randn(batch_size, 150, 3, 224, 224)
audio = torch.randn(batch_size, 220500)
sensors = torch.randn(batch_size, 15)

# Forward
logits = model(video, audio, sensors)

# Predição
predicted_class, confidence = model.predict(video, audio, sensors)

print(f"Classe: {predicted_class.item()}")
print(f"Confiança: {confidence.item():.2%}")
```

### Inferência com Ensemble

```python
from ml.ensemble import EnsembleVariants

# Criar ensemble
ensemble = EnsembleVariants.fast_ensemble()
ensemble.eval()

# Predição com detalhes
with torch.no_grad():
    result = ensemble.predict(video, audio, sensors, return_details=True)

print(f"Classe: {result.predicted_class}")
print(f"Confiança: {result.confidence:.2%}")
print(f"Método: {result.voting_method}")

# Predições individuais
for pred in result.individual_predictions:
    print(f"Modelo {pred['model_index']}: "
          f"classe={pred['predicted_class']}, "
          f"conf={pred['confidence']:.2%}")
```

### Ensemble Personalizado

```python
from ml.models import FusionVariants
from ml.ensemble import NeuralEnsemble

# Criar modelos específicos
models = [
    FusionVariants.lightweight(),
    FusionVariants.balanced(),
    FusionVariants.audio_focused()
]

# Criar ensemble com pesos customizados
ensemble = NeuralEnsemble(
    models=models,
    method='weighted',
    weights=[0.2, 0.5, 0.3]  # Dar mais peso ao balanced
)

# Uso
result = ensemble.predict(video, audio, sensors)
```

---

## 📈 Comparação de Performance

| Configuração | Params | Speed | Accuracy* | Uso Recomendado |
|--------------|--------|-------|-----------|-----------------|
| **Lightweight** | 8M | 50ms | 75% | Tempo real, edge |
| **Balanced** | 12M | 100ms | 82% | Padrão |
| **Accurate** | 25M | 200ms | 88% | Offline, melhor accuracy |
| **Fast Ensemble** | 24M | 150ms | 85% | Tempo real + ensemble |
| **Accurate Ensemble** | 37M | 300ms | 90% | Batch processing |
| **Full Ensemble** | 70M | 500ms | 93% | Máxima performance |

\* Valores estimados, dependem do dataset

---

## 🎓 Quando Usar Cada Abordagem

### Modelo Único

**Use quando**:
- ✅ Inferência em tempo real estrito (< 100ms)
- ✅ Hardware limitado (mobile, edge)
- ✅ Dataset pequeno (overfitting com ensemble)

**Recomendado**: `FusionVariants.balanced()`

### Ensemble Pequeno (2-3 modelos)

**Use quando**:
- ✅ Quer melhorar accuracy sem muito overhead
- ✅ Tem GPU decent (RTX 3060+)
- ✅ Latência aceitável (< 200ms)

**Recomendado**: `EnsembleVariants.fast_ensemble()`

### Ensemble Completo (5+ modelos)

**Use quando**:
- ✅ Accuracy é crítica
- ✅ Processamento em batch (não tempo real)
- ✅ Tem GPU poderosa (RTX 4090)

**Recomendado**: `EnsembleVariants.full_ensemble()`

---

## 🔧 Configuração para Produção

### Config para RTX 4090

```python
# config/ml_settings.py

ML_CONFIG = {
    # Produção: Accurate Ensemble
    'model_type': 'accurate_ensemble',
    'batch_size': 16,
    'use_fp16': True,  # Mixed precision
    'num_workers': 4,

    # Fallback: Se latência > 300ms, usar Fast Ensemble
    'fallback_model': 'fast_ensemble',
    'max_latency_ms': 300
}
```

### Otimizações

```python
# 1. Mixed Precision (FP16)
model = model.half()  # 2x mais rápido, metade da memória

# 2. TorchScript (JIT)
model_scripted = torch.jit.script(model)

# 3. ONNX (para deploy em outras plataformas)
torch.onnx.export(model, (video, audio, sensors), "model.onnx")

# 4. TensorRT (NVIDIA otimizado)
# Converta ONNX para TensorRT para máxima velocidade
```

---

## 📚 Referências

- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [ResNet](https://arxiv.org/abs/1512.03385)
- [Vision Transformer](https://arxiv.org/abs/2010.11929)
- [Multimodal Fusion](https://arxiv.org/abs/2103.05561)
- [Ensemble Methods](https://arxiv.org/abs/1404.3230)

---

**Última Atualização**: 2025-01-17
**Versão**: 1.0.0
**Hardware Target**: NVIDIA RTX 4090
