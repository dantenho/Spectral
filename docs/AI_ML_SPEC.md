# 🧠 Especificação de IA/ML - Pipeline de Treinamento

## 🎯 Objetivo

Desenvolver um modelo de **fusão multimodal** (vídeo + áudio + sensores) para classificar eventos anômalos detectados pelo sistema Spectral, utilizando PyTorch e treinamento em GPU NVIDIA RTX 4090.

---

## 📊 Classes de Classificação

O modelo classificará eventos em 4 categorias:

| Classe | Descrição | Características |
|--------|-----------|-----------------|
| **Ruído_Ambiente** | Ruído natural sem correlação | Baixa correlação entre sensores |
| **Interferência_Eletrônica** | Dispositivos eletrônicos | Anomalia magnética + áudio eletrônico |
| **Anomalia_Correlacionada** | Evento de interesse | Alta correlação multi-sensorial |
| **Forma_Humanoide_Potencial** | Detecção de forma + anomalia | Humanoid flag + correlação |

---

## 🏗️ Arquitetura do Modelo

### Visão Geral

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT                                    │
├─────────────────────┬───────────────────┬───────────────────────┤
│   Video Frames      │   Audio Waveform  │   Sensor Data         │
│   [N, 3, 224, 224]  │   [1, 220500]     │   [10, features]      │
└──────────┬──────────┴──────────┬────────┴──────────┬────────────┘
           │                     │                   │
           ▼                     ▼                   ▼
┌──────────────────┐  ┌────────────────┐  ┌──────────────────┐
│ Video Encoder    │  │ Audio Encoder  │  │ Sensor MLP       │
│ (EfficientNet-B0)│  │ (1D CNN)       │  │                  │
│                  │  │                │  │                  │
│ Pre-trained      │  │ Custom         │  │ Custom           │
│ ImageNet         │  │ Spectrogram    │  │                  │
└────────┬─────────┘  └────────┬───────┘  └────────┬─────────┘
         │                     │                   │
         │ [1280]              │ [256]             │ [64]
         │                     │                   │
         └─────────────────────┴───────────────────┘
                               │
                               ▼
                    ┌──────────────────┐
                    │  Concatenation   │
                    │  [1280+256+64]   │
                    │  = [1600]        │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Fusion MLP      │
                    │  [1600→512→128]  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  Classifier      │
                    │  [128→4]         │
                    │  Softmax         │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  OUTPUT          │
                    │  [4 classes]     │
                    │  + Confidence    │
                    └──────────────────┘
```

---

## 🎬 1. Video Encoder

### Arquitetura

Utilizamos **EfficientNet-B0** pré-treinado no ImageNet:
- Parâmetros: ~5.3M
- Velocidade: ~35 FPS em RTX 4090
- Output: 1280-dim embedding

### Implementação

```python
# ml/models/video_encoder.py

import torch
import torch.nn as nn
import timm

class VideoEncoder(nn.Module):
    """
    Encoder de vídeo usando EfficientNet-B0 pré-treinado
    Input: [N, 3, 224, 224] onde N é o número de frames
    Output: [1280] embedding do vídeo completo
    """

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()

        # EfficientNet-B0 do timm
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )

        # Freeze backbone se especificado
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.embedding_dim = 1280

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_frames: [batch_size, num_frames, 3, 224, 224]

        Returns:
            embeddings: [batch_size, 1280]
        """
        batch_size, num_frames, C, H, W = video_frames.shape

        # Reshape para processar todos os frames de uma vez
        frames = video_frames.view(batch_size * num_frames, C, H, W)

        # Extrair features de cada frame
        frame_features = self.backbone(frames)  # [batch*num_frames, 1280]

        # Reshape de volta
        frame_features = frame_features.view(batch_size, num_frames, -1)

        # Agregar frames: média temporal
        video_embedding = torch.mean(frame_features, dim=1)  # [batch, 1280]

        return video_embedding


# Teste
if __name__ == "__main__":
    encoder = VideoEncoder(pretrained=True)
    video = torch.randn(2, 150, 3, 224, 224)  # 2 videos, 150 frames (5s @ 30fps)
    embedding = encoder(video)
    print(f"Video embedding shape: {embedding.shape}")  # [2, 1280]
```

---

## 🎵 2. Audio Encoder

### Arquitetura

CNN 1D sobre espectrograma Mel:
- Input: Mel-spectrogram [128, 431] (5 segundos)
- Layers: 4x Conv2D + BatchNorm + MaxPool
- Output: 256-dim embedding

### Implementação

```python
# ml/models/audio_encoder.py

import torch
import torch.nn as nn
import torchaudio

class AudioEncoder(nn.Module):
    """
    Encoder de áudio usando CNN sobre Mel-spectrogram
    Input: Audio waveform [batch, 220500] (5s @ 44.1kHz)
    Output: [256] embedding de áudio
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        n_mels: int = 128,
        embedding_dim: int = 256
    ):
        super().__init__()

        self.sample_rate = sample_rate

        # Mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=512
        )

        # Log-scale
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # CNN Layers
        self.conv_layers = nn.Sequential(
            # Conv1: [1, 128, 431] -> [32, 64, 215]
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv2: [32, 64, 215] -> [64, 32, 107]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv3: [64, 32, 107] -> [128, 16, 53]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv4: [128, 16, 53] -> [256, 8, 26]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.embedding_dim = embedding_dim

        # Projection head
        self.projection = nn.Linear(256, embedding_dim)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [batch_size, samples] - Audio waveform

        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        # Mel-spectrogram
        mel_spec = self.mel_transform(waveform)  # [batch, n_mels, time]
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # Add channel dimension
        mel_spec_db = mel_spec_db.unsqueeze(1)  # [batch, 1, n_mels, time]

        # CNN
        features = self.conv_layers(mel_spec_db)  # [batch, 256, 8, 26]

        # Global pooling
        pooled = self.global_pool(features)  # [batch, 256, 1, 1]
        pooled = pooled.squeeze(-1).squeeze(-1)  # [batch, 256]

        # Projection
        embedding = self.projection(pooled)  # [batch, embedding_dim]

        return embedding


# Teste
if __name__ == "__main__":
    encoder = AudioEncoder(embedding_dim=256)
    audio = torch.randn(2, 220500)  # 2 audios, 5 segundos @ 44.1kHz
    embedding = encoder(audio)
    print(f"Audio embedding shape: {embedding.shape}")  # [2, 256]
```

---

## 📊 3. Sensor MLP

### Arquitetura

Rede totalmente conectada para processar dados tabulares:
- Input: Features dos sensores (10 timesteps × features)
- Layers: 2x Linear + ReLU + Dropout
- Output: 64-dim embedding

### Features de Entrada

Para cada evento (janela de 5 segundos):
- Magnetic: mean, std, max, min magnitude
- Accelerometer: mean norm
- Gyroscope: mean norm
- Orientation: final yaw (direção)
- Audio: peak amplitude
- Flags: humanoid_detected, bluetooth_count

Total: **~15 features**

### Implementação

```python
# ml/models/sensor_encoder.py

import torch
import torch.nn as nn

class SensorEncoder(nn.Module):
    """
    MLP para processar dados tabulares de sensores
    Input: [batch, num_features]
    Output: [batch, 64]
    """

    def __init__(self, input_dim: int = 15, embedding_dim: int = 64):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, embedding_dim),
        )

        self.embedding_dim = embedding_dim

    def forward(self, sensor_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_features: [batch_size, num_features]

        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        return self.mlp(sensor_features)
```

---

## 🔗 4. Fusion Classifier

### Arquitetura

Combina todos os embeddings e classifica:
- Input: [1280 + 256 + 64] = 1600
- Hidden: [512, 128]
- Output: 4 classes

### Implementação

```python
# ml/models/fusion_classifier.py

import torch
import torch.nn as nn
from .video_encoder import VideoEncoder
from .audio_encoder import AudioEncoder
from .sensor_encoder import SensorEncoder

class SpectralFusionModel(nn.Module):
    """
    Modelo completo de fusão multimodal
    """

    def __init__(
        self,
        num_classes: int = 4,
        video_pretrained: bool = True,
        freeze_video: bool = True
    ):
        super().__init__()

        # Encoders
        self.video_encoder = VideoEncoder(
            pretrained=video_pretrained,
            freeze_backbone=freeze_video
        )
        self.audio_encoder = AudioEncoder(embedding_dim=256)
        self.sensor_encoder = SensorEncoder(input_dim=15, embedding_dim=64)

        # Dimensões
        fusion_input_dim = (
            self.video_encoder.embedding_dim +
            self.audio_encoder.embedding_dim +
            self.sensor_encoder.embedding_dim
        )  # 1280 + 256 + 64 = 1600

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Classifier head
        self.classifier = nn.Linear(128, num_classes)

    def forward(
        self,
        video_frames: torch.Tensor,
        audio_waveform: torch.Tensor,
        sensor_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            video_frames: [batch, num_frames, 3, 224, 224]
            audio_waveform: [batch, samples]
            sensor_features: [batch, num_features]

        Returns:
            logits: [batch, num_classes]
        """
        # Encode cada modalidade
        video_emb = self.video_encoder(video_frames)      # [batch, 1280]
        audio_emb = self.audio_encoder(audio_waveform)    # [batch, 256]
        sensor_emb = self.sensor_encoder(sensor_features) # [batch, 64]

        # Concatenar
        fused = torch.cat([video_emb, audio_emb, sensor_emb], dim=1)  # [batch, 1600]

        # Fusion
        fused_features = self.fusion_mlp(fused)  # [batch, 128]

        # Classificação
        logits = self.classifier(fused_features)  # [batch, num_classes]

        return logits

    def predict(
        self,
        video_frames: torch.Tensor,
        audio_waveform: torch.Tensor,
        sensor_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predição com softmax

        Returns:
            (class_indices, confidences)
        """
        logits = self.forward(video_frames, audio_waveform, sensor_features)
        probabilities = torch.softmax(logits, dim=1)

        confidences, predicted_classes = torch.max(probabilities, dim=1)

        return predicted_classes, confidences
```

---

## 📦 5. Dataset

### Implementação

```python
# ml/dataset.py

import torch
from torch.utils.data import Dataset
import cv2
import soundfile as sf
import pandas as pd
import json
from pathlib import Path
from typing import Tuple

class SpectralDataset(Dataset):
    """
    Dataset para carregar eventos salvos
    """

    def __init__(
        self,
        events_dir: Path,
        transform=None,
        num_frames: int = 150  # 5s @ 30fps
    ):
        self.events_dir = Path(events_dir)
        self.transform = transform
        self.num_frames = num_frames

        # Descobrir todos os eventos
        self.event_paths = sorted(list(self.events_dir.glob("event_*")))

        # Mapeamento de classes
        self.class_to_idx = {
            "Ruído_Ambiente": 0,
            "Interferência_Eletrônica": 1,
            "Anomalia_Correlacionada": 2,
            "Forma_Humanoide_Potencial": 3
        }

    def __len__(self) -> int:
        return len(self.event_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        event_path = self.event_paths[idx]

        # Carregar metadata
        with open(event_path / "metadata.json") as f:
            metadata = json.load(f)

        # Classe
        classification = metadata.get("classification", "Ruído_Ambiente")
        label = self.class_to_idx[classification]

        # Vídeo
        video_frames = self._load_video(event_path / "video.mp4")

        # Áudio
        audio_waveform = self._load_audio(event_path / "audio.wav")

        # Sensores
        sensor_features = self._load_sensors(event_path / "sensors.csv")

        return video_frames, audio_waveform, sensor_features, label

    def _load_video(self, video_path: Path) -> torch.Tensor:
        """Carrega vídeo e extrai frames uniformemente"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Selecionar frames uniformemente
        frame_indices = torch.linspace(0, total_frames - 1, self.num_frames).long()

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx.item())
            ret, frame = cap.read()
            if ret:
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize para 224x224
                frame = cv2.resize(frame, (224, 224))
                # Normalizar
                frame = frame.astype('float32') / 255.0
                frames.append(frame)

        cap.release()

        # [num_frames, H, W, C] -> [num_frames, C, H, W]
        frames = torch.tensor(frames).permute(0, 3, 1, 2)

        return frames

    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """Carrega áudio WAV"""
        waveform, sr = sf.read(str(audio_path))

        # Mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Normalizar
        waveform = waveform.astype('float32')

        # Garantir 5 segundos (220500 samples @ 44.1kHz)
        target_length = 220500
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        elif len(waveform) < target_length:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))

        return torch.tensor(waveform)

    def _load_sensors(self, sensor_path: Path) -> torch.Tensor:
        """Carrega e processa dados de sensores"""
        df = pd.read_csv(sensor_path)

        # Extrair features estatísticas
        features = []

        # Magnetic
        if 'magnetic_magnitude' in df.columns:
            features.extend([
                df['magnetic_magnitude'].mean(),
                df['magnetic_magnitude'].std(),
                df['magnetic_magnitude'].max(),
                df['magnetic_magnitude'].min()
            ])

        # Accelerometer (compute norm)
        if all(col in df.columns for col in ['accel_x', 'accel_y', 'accel_z']):
            accel_norm = (df['accel_x']**2 + df['accel_y']**2 + df['accel_z']**2)**0.5
            features.append(accel_norm.mean())

        # Gyroscope (compute norm)
        if all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
            gyro_norm = (df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)**0.5
            features.append(gyro_norm.mean())

        # Orientation (final yaw)
        if 'orientation_yaw' in df.columns:
            features.append(df['orientation_yaw'].iloc[-1])

        # Audio peak
        if 'audio_peak' in df.columns:
            features.append(df['audio_peak'].max())

        # Boolean flags
        if 'humanoid_detected' in df.columns:
            features.append(float(df['humanoid_detected'].any()))

        if 'bluetooth_count' in df.columns:
            features.append(df['bluetooth_count'].mean())

        return torch.tensor(features, dtype=torch.float32)
```

---

## 🎓 6. Training Loop (PyTorch Lightning)

```python
# ml/training.py

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, ConfusionMatrix
import wandb

from .models.fusion_classifier import SpectralFusionModel
from .dataset import SpectralDataset

class SpectralLightningModule(pl.LightningModule):
    """
    Lightning Module para treinamento do modelo Spectral
    """

    def __init__(
        self,
        num_classes: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()

        # Modelo
        self.model = SpectralFusionModel(
            num_classes=num_classes,
            video_pretrained=True,
            freeze_video=True
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss()

        # Métricas
        self.train_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = Accuracy(task='multiclass', num_classes=num_classes)
        self.f1_score = F1Score(task='multiclass', num_classes=num_classes, average='macro')

    def forward(self, video, audio, sensors):
        return self.model(video, audio, sensors)

    def training_step(self, batch, batch_idx):
        video, audio, sensors, labels = batch

        # Forward
        logits = self(video, audio, sensors)

        # Loss
        loss = self.criterion(logits, labels)

        # Accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, labels)

        # Log
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        video, audio, sensors, labels = batch

        # Forward
        logits = self(video, audio, sensors)

        # Loss
        loss = self.criterion(logits, labels)

        # Métricas
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, labels)
        f1 = self.f1_score(preds, labels)

        # Log
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_epoch=True, prog_bar=True)
        self.log('val_f1', f1, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


# Script de treinamento
def train_model():
    # Dataset
    train_dataset = SpectralDataset(Path("data/training/train"))
    val_dataset = SpectralDataset(Path("data/training/val"))

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Modelo
    model = SpectralLightningModule(learning_rate=1e-3)

    # Wandb Logger
    wandb_logger = pl.loggers.WandbLogger(
        project="spectral",
        name="fusion_model_v1"
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator='gpu',
        devices=1,
        logger=wandb_logger,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath='models/checkpoints',
                filename='spectral-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',
                mode='min',
                save_top_k=3
            ),
            pl.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            )
        ]
    )

    # Treinar
    trainer.fit(model, train_loader, val_loader)
```

---

## 📈 Performance Esperada

### Métricas Target (após 50 epochs)

| Métrica | Target | Fase MVP |
|---------|--------|----------|
| **Validation Accuracy** | > 85% | > 70% |
| **F1-Score (Macro)** | > 0.80 | > 0.65 |
| **Inference Time** | < 200ms | < 500ms |
| **Model Size** | < 100 MB | - |

---

## 🚀 Pipeline de Deploy

```python
# ml/inference.py

import torch
from pathlib import Path
from .models.fusion_classifier import SpectralFusionModel

class SpectralInference:
    """Inference wrapper para produção"""

    def __init__(self, checkpoint_path: Path, device: str = 'cuda'):
        self.device = device
        self.model = SpectralFusionModel.load_from_checkpoint(str(checkpoint_path))
        self.model.to(device)
        self.model.eval()

        self.class_names = [
            "Ruído_Ambiente",
            "Interferência_Eletrônica",
            "Anomalia_Correlacionada",
            "Forma_Humanoide_Potencial"
        ]

    @torch.no_grad()
    def predict(self, video, audio, sensors):
        """Predição em um único evento"""
        # Adicionar batch dimension
        video = video.unsqueeze(0).to(self.device)
        audio = audio.unsqueeze(0).to(self.device)
        sensors = sensors.unsqueeze(0).to(self.device)

        # Forward
        logits = self.model(video, audio, sensors)
        probabilities = torch.softmax(logits, dim=1)[0]

        # Top prediction
        confidence, predicted_class = torch.max(probabilities, dim=0)

        return {
            'class': self.class_names[predicted_class],
            'confidence': confidence.item(),
            'probabilities': {
                name: prob.item()
                for name, prob in zip(self.class_names, probabilities)
            }
        }
```

---

**Última Atualização**: 2025-01-17
**Versão**: 1.0.0
**Framework**: PyTorch 2.1 + Lightning 2.1
