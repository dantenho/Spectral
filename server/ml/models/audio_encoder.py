"""
Audio Encoder - Múltiplas Arquiteturas

Implementa diferentes arquiteturas para extração de features de áudio
"""

import torch
import torch.nn as nn
import torchaudio
from typing import Literal, Optional


class AudioEncoder(nn.Module):
    """
    Encoder de áudio base com arquitetura configurável

    Processa espectrograma Mel com CNN ou Transformer
    """

    def __init__(
        self,
        architecture: Literal['cnn', 'resnet', 'transformer'] = 'cnn',
        sample_rate: int = 44100,
        n_fft: int = 2048,
        n_mels: int = 128,
        embedding_dim: int = 256,
        dropout: float = 0.3
    ):
        """
        Args:
            architecture: Tipo de arquitetura
            sample_rate: Taxa de amostragem
            n_fft: Tamanho da FFT
            n_mels: Número de bandas Mel
            embedding_dim: Dimensão do embedding final
            dropout: Taxa de dropout
        """
        super().__init__()

        self.architecture = architecture
        self.sample_rate = sample_rate
        self.embedding_dim = embedding_dim

        # Mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=512,
            power=2.0
        )

        # Log-scale
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

        # Arquitetura específica
        if architecture == 'cnn':
            self.encoder = self._build_cnn_encoder(n_mels, embedding_dim, dropout)
        elif architecture == 'resnet':
            self.encoder = self._build_resnet_encoder(n_mels, embedding_dim, dropout)
        elif architecture == 'transformer':
            self.encoder = self._build_transformer_encoder(n_mels, embedding_dim, dropout)

    def _build_cnn_encoder(self, n_mels: int, embedding_dim: int, dropout: float) -> nn.Module:
        """CNN simples e eficiente"""
        return nn.Sequential(
            # Conv1: [1, 128, T] -> [32, 64, T/2]
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            # Conv2: [32, 64, T/2] -> [64, 32, T/4]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            # Conv3: [64, 32, T/4] -> [128, 16, T/8]
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            # Conv4: [128, 16, T/8] -> [256, 8, T/16]
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            # Global pooling e projection
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def _build_resnet_encoder(self, n_mels: int, embedding_dim: int, dropout: float) -> nn.Module:
        """ResNet-style com blocos residuais"""

        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
                self.bn2 = nn.BatchNorm2d(out_channels)

                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride),
                        nn.BatchNorm2d(out_channels)
                    )

            def forward(self, x):
                out = torch.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = torch.relu(out)
                return out

        return nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),

            ResidualBlock(64, 64),
            ResidualBlock(64, 128, 2),
            ResidualBlock(128, 256, 2),
            ResidualBlock(256, 512, 2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def _build_transformer_encoder(self, n_mels: int, embedding_dim: int, dropout: float) -> nn.Module:
        """Transformer para capturar dependências temporais"""

        class TransformerEncoder(nn.Module):
            def __init__(self, n_mels, embedding_dim, dropout):
                super().__init__()

                self.patch_embedding = nn.Conv2d(1, embedding_dim, kernel_size=(n_mels // 8, 4), stride=(n_mels // 8, 4))

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=8,
                    dim_feedforward=embedding_dim * 4,
                    dropout=dropout,
                    batch_first=True
                )

                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

                self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
                self.fc = nn.Linear(embedding_dim, embedding_dim)

            def forward(self, x):
                # Patch embedding
                x = self.patch_embedding(x)  # [B, E, H', W']
                B, E, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)  # [B, H'*W', E]

                # Add CLS token
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = torch.cat([cls_tokens, x], dim=1)

                # Transformer
                x = self.transformer(x)

                # Use CLS token
                x = x[:, 0]
                x = self.fc(x)

                return x

        return TransformerEncoder(n_mels, embedding_dim, dropout)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [batch_size, samples]

        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        # Mel-spectrogram
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        # Adicionar dimensão de canal
        mel_spec_db = mel_spec_db.unsqueeze(1)  # [B, 1, n_mels, T]

        # Encoder
        embedding = self.encoder(mel_spec_db)

        return embedding


# ============================================================================
# VARIANTES DE AUDIO ENCODER
# ============================================================================

class AudioEncoderVariants:
    """Factory para criar diferentes variantes de Audio Encoder"""

    @staticmethod
    def cnn_small(embedding_dim: int = 256, **kwargs) -> AudioEncoder:
        """CNN pequeno e rápido"""
        return AudioEncoder(
            architecture='cnn',
            n_mels=64,
            embedding_dim=embedding_dim,
            dropout=0.2,
            **kwargs
        )

    @staticmethod
    def cnn_large(embedding_dim: int = 512, **kwargs) -> AudioEncoder:
        """CNN grande para melhor accuracy"""
        return AudioEncoder(
            architecture='cnn',
            n_mels=128,
            embedding_dim=embedding_dim,
            dropout=0.3,
            **kwargs
        )

    @staticmethod
    def resnet(embedding_dim: int = 256, **kwargs) -> AudioEncoder:
        """ResNet com blocos residuais"""
        return AudioEncoder(
            architecture='resnet',
            n_mels=128,
            embedding_dim=embedding_dim,
            dropout=0.3,
            **kwargs
        )

    @staticmethod
    def transformer(embedding_dim: int = 256, **kwargs) -> AudioEncoder:
        """Transformer para dependências temporais"""
        return AudioEncoder(
            architecture='transformer',
            n_mels=128,
            embedding_dim=embedding_dim,
            dropout=0.1,
            **kwargs
        )


# ============================================================================
# TESTE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DE AUDIO ENCODERS")
    print("=" * 60)

    # Criar áudio de teste (5 segundos @ 44.1kHz)
    batch_size = 2
    duration = 5
    sample_rate = 44100
    audio = torch.randn(batch_size, sample_rate * duration)

    print(f"\nInput shape: {audio.shape}")
    print(f"Duration: {duration}s")

    # Testar diferentes variantes
    variants = [
        ('CNN Small', AudioEncoderVariants.cnn_small()),
        ('CNN Large', AudioEncoderVariants.cnn_large()),
        ('ResNet', AudioEncoderVariants.resnet()),
        ('Transformer', AudioEncoderVariants.transformer()),
    ]

    for name, encoder in variants:
        encoder.eval()
        with torch.no_grad():
            embedding = encoder(audio)

        print(f"\n{name}:")
        print(f"  Output shape: {embedding.shape}")
        print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        print(f"  Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")

    # Benchmark de velocidade
    print("\n" + "=" * 60)
    print("BENCHMARK DE VELOCIDADE")
    print("=" * 60)

    import time

    encoder = AudioEncoderVariants.cnn_small()
    encoder.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = encoder(audio)

    # Benchmark
    num_iterations = 100
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = encoder(audio)

    elapsed_time = time.time() - start_time
    avg_time = (elapsed_time / num_iterations) * 1000

    print(f"\nCNN Small:")
    print(f"  Avg time: {avg_time:.2f} ms")
    print(f"  Throughput: {num_iterations / elapsed_time:.1f} samples/sec")
