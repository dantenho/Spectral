"""
Fusion Classifier - Combina múltiplos encoders

Implementa diferentes estratégias de fusão multimodal
"""

import torch
import torch.nn as nn
from typing import Literal, Optional

from .video_encoder import VideoEncoder, VideoEncoderVariants
from .audio_encoder import AudioEncoder, AudioEncoderVariants
from .sensor_encoder import SensorEncoder, SensorEncoderVariants


class FusionClassifier(nn.Module):
    """
    Classificador que funde embeddings de vídeo, áudio e sensores

    Suporta diferentes estratégias de fusão
    """

    def __init__(
        self,
        video_encoder: VideoEncoder,
        audio_encoder: AudioEncoder,
        sensor_encoder: SensorEncoder,
        num_classes: int = 4,
        fusion_strategy: Literal['concat', 'attention', 'gated', 'bilinear'] = 'concat',
        dropout: float = 0.5
    ):
        """
        Args:
            video_encoder: Encoder de vídeo
            audio_encoder: Encoder de áudio
            sensor_encoder: Encoder de sensores
            num_classes: Número de classes
            fusion_strategy: Estratégia de fusão
            dropout: Taxa de dropout
        """
        super().__init__()

        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.sensor_encoder = sensor_encoder

        self.fusion_strategy = fusion_strategy
        self.num_classes = num_classes

        # Dimensões dos embeddings
        video_dim = video_encoder.final_dim
        audio_dim = audio_encoder.embedding_dim
        sensor_dim = sensor_encoder.embedding_dim

        # Construir camada de fusão baseado na estratégia
        if fusion_strategy == 'concat':
            self.fusion_layer = self._build_concat_fusion(
                video_dim, audio_dim, sensor_dim, num_classes, dropout
            )

        elif fusion_strategy == 'attention':
            self.fusion_layer = self._build_attention_fusion(
                video_dim, audio_dim, sensor_dim, num_classes, dropout
            )

        elif fusion_strategy == 'gated':
            self.fusion_layer = self._build_gated_fusion(
                video_dim, audio_dim, sensor_dim, num_classes, dropout
            )

        elif fusion_strategy == 'bilinear':
            self.fusion_layer = self._build_bilinear_fusion(
                video_dim, audio_dim, sensor_dim, num_classes, dropout
            )

    def _build_concat_fusion(
        self, v_dim: int, a_dim: int, s_dim: int, num_classes: int, dropout: float
    ) -> nn.Module:
        """Concatenação simples"""
        total_dim = v_dim + a_dim + s_dim

        return nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout * 0.6),

            nn.Linear(128, num_classes)
        )

    def _build_attention_fusion(
        self, v_dim: int, a_dim: int, s_dim: int, num_classes: int, dropout: float
    ) -> nn.Module:
        """Fusão com atenção entre modalidades"""

        class AttentionFusion(nn.Module):
            def __init__(self, v_dim, a_dim, s_dim, num_classes, dropout):
                super().__init__()

                # Projetar todas as modalidades para mesma dimensão
                self.common_dim = 256

                self.video_proj = nn.Linear(v_dim, self.common_dim)
                self.audio_proj = nn.Linear(a_dim, self.common_dim)
                self.sensor_proj = nn.Linear(s_dim, self.common_dim)

                # Multi-head attention
                self.attention = nn.MultiheadAttention(
                    embed_dim=self.common_dim,
                    num_heads=8,
                    dropout=dropout,
                    batch_first=True
                )

                # Classifier
                self.classifier = nn.Sequential(
                    nn.Linear(self.common_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(128, num_classes)
                )

            def forward(self, video_emb, audio_emb, sensor_emb):
                # Projetar
                v = self.video_proj(video_emb).unsqueeze(1)  # [B, 1, D]
                a = self.audio_proj(audio_emb).unsqueeze(1)
                s = self.sensor_proj(sensor_emb).unsqueeze(1)

                # Concatenar como sequência
                x = torch.cat([v, a, s], dim=1)  # [B, 3, D]

                # Self-attention
                x, _ = self.attention(x, x, x)

                # Pooling
                x = torch.mean(x, dim=1)  # [B, D]

                # Classificar
                return self.classifier(x)

        return AttentionFusion(v_dim, a_dim, s_dim, num_classes, dropout)

    def _build_gated_fusion(
        self, v_dim: int, a_dim: int, s_dim: int, num_classes: int, dropout: float
    ) -> nn.Module:
        """Fusão com gating (aprende pesos para cada modalidade)"""

        class GatedFusion(nn.Module):
            def __init__(self, v_dim, a_dim, s_dim, num_classes, dropout):
                super().__init__()

                # Gates para cada modalidade
                self.video_gate = nn.Sequential(
                    nn.Linear(v_dim, v_dim),
                    nn.Sigmoid()
                )

                self.audio_gate = nn.Sequential(
                    nn.Linear(a_dim, a_dim),
                    nn.Sigmoid()
                )

                self.sensor_gate = nn.Sequential(
                    nn.Linear(s_dim, s_dim),
                    nn.Sigmoid()
                )

                # Fusion MLP
                total_dim = v_dim + a_dim + s_dim

                self.mlp = nn.Sequential(
                    nn.Linear(total_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout),

                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.6),

                    nn.Linear(128, num_classes)
                )

            def forward(self, video_emb, audio_emb, sensor_emb):
                # Aplicar gates (modular importância)
                v_gated = video_emb * self.video_gate(video_emb)
                a_gated = audio_emb * self.audio_gate(audio_emb)
                s_gated = sensor_emb * self.sensor_gate(sensor_emb)

                # Concatenar
                fused = torch.cat([v_gated, a_gated, s_gated], dim=1)

                # Classificar
                return self.mlp(fused)

        return GatedFusion(v_dim, a_dim, s_dim, num_classes, dropout)

    def _build_bilinear_fusion(
        self, v_dim: int, a_dim: int, s_dim: int, num_classes: int, dropout: float
    ) -> nn.Module:
        """Fusão bilinear (captura interações de segunda ordem)"""

        class BilinearFusion(nn.Module):
            def __init__(self, v_dim, a_dim, s_dim, num_classes, dropout):
                super().__init__()

                # Bilinear pooling entre pares
                self.bilinear_va = nn.Bilinear(v_dim, a_dim, 256)
                self.bilinear_vs = nn.Bilinear(v_dim, s_dim, 256)
                self.bilinear_as = nn.Bilinear(a_dim, s_dim, 256)

                # Classifier
                self.classifier = nn.Sequential(
                    nn.Linear(256 * 3, 512),
                    nn.ReLU(),
                    nn.Dropout(dropout),

                    nn.Linear(512, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout * 0.6),

                    nn.Linear(128, num_classes)
                )

            def forward(self, video_emb, audio_emb, sensor_emb):
                # Interações bi lineares
                va = self.bilinear_va(video_emb, audio_emb)
                vs = self.bilinear_vs(video_emb, sensor_emb)
                as_interaction = self.bilinear_as(audio_emb, sensor_emb)

                # Concatenar
                fused = torch.cat([va, vs, as_interaction], dim=1)

                # Classificar
                return self.classifier(fused)

        return BilinearFusion(v_dim, a_dim, s_dim, num_classes, dropout)

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        sensors: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            video: [batch, num_frames, 3, H, W]
            audio: [batch, samples]
            sensors: [batch, features]

        Returns:
            logits: [batch, num_classes]
        """
        # Encode cada modalidade
        video_emb = self.video_encoder(video)
        audio_emb = self.audio_encoder(audio)
        sensor_emb = self.sensor_encoder(sensors)

        # Fusão
        if self.fusion_strategy == 'concat':
            fused = torch.cat([video_emb, audio_emb, sensor_emb], dim=1)
            logits = self.fusion_layer(fused)
        else:
            logits = self.fusion_layer(video_emb, audio_emb, sensor_emb)

        return logits

    def predict(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        sensors: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predição com softmax

        Returns:
            (predicted_classes, confidences)
        """
        logits = self.forward(video, audio, sensors)
        probabilities = torch.softmax(logits, dim=1)

        confidences, predicted_classes = torch.max(probabilities, dim=1)

        return predicted_classes, confidences


# ============================================================================
# VARIANTES PRÉ-CONFIGURADAS
# ============================================================================

class FusionVariants:
    """Factory para criar diferentes combinações de modelos"""

    @staticmethod
    def lightweight(num_classes: int = 4) -> FusionClassifier:
        """Modelo leve e rápido"""
        return FusionClassifier(
            video_encoder=VideoEncoderVariants.mobilenetv3_large(
                num_frames=150,
                temporal_aggregation='mean'
            ),
            audio_encoder=AudioEncoderVariants.cnn_small(embedding_dim=256),
            sensor_encoder=SensorEncoderVariants.simple(embedding_dim=64),
            num_classes=num_classes,
            fusion_strategy='concat',
            dropout=0.3
        )

    @staticmethod
    def balanced(num_classes: int = 4) -> FusionClassifier:
        """Balanço entre velocidade e accuracy"""
        return FusionClassifier(
            video_encoder=VideoEncoderVariants.efficientnet_b0(
                num_frames=150,
                temporal_aggregation='lstm'
            ),
            audio_encoder=AudioEncoderVariants.resnet(embedding_dim=256),
            sensor_encoder=SensorEncoderVariants.deep(embedding_dim=64),
            num_classes=num_classes,
            fusion_strategy='attention',
            dropout=0.4
        )

    @staticmethod
    def accurate(num_classes: int = 4) -> FusionClassifier:
        """Máxima accuracy"""
        return FusionClassifier(
            video_encoder=VideoEncoderVariants.efficientnet_b2(
                num_frames=150,
                temporal_aggregation='attention'
            ),
            audio_encoder=AudioEncoderVariants.transformer(embedding_dim=512),
            sensor_encoder=SensorEncoderVariants.attention(embedding_dim=128),
            num_classes=num_classes,
            fusion_strategy='bilinear',
            dropout=0.5
        )

    @staticmethod
    def vision_focused(num_classes: int = 4) -> FusionClassifier:
        """Foco em vídeo/pose estimation"""
        return FusionClassifier(
            video_encoder=VideoEncoderVariants.convnext_tiny(
                num_frames=150,
                temporal_aggregation='attention'
            ),
            audio_encoder=AudioEncoderVariants.cnn_small(embedding_dim=128),
            sensor_encoder=SensorEncoderVariants.simple(embedding_dim=32),
            num_classes=num_classes,
            fusion_strategy='gated',
            dropout=0.4
        )

    @staticmethod
    def audio_focused(num_classes: int = 4) -> FusionClassifier:
        """Foco em análise de áudio/EVP"""
        return FusionClassifier(
            video_encoder=VideoEncoderVariants.mobilenetv3_large(
                num_frames=150,
                temporal_aggregation='mean'
            ),
            audio_encoder=AudioEncoderVariants.cnn_large(embedding_dim=512),
            sensor_encoder=SensorEncoderVariants.simple(embedding_dim=64),
            num_classes=num_classes,
            fusion_strategy='attention',
            dropout=0.3
        )


# ============================================================================
# TESTE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTE DE FUSION CLASSIFIERS")
    print("=" * 70)

    # Criar dados de teste
    batch_size = 2
    video = torch.randn(batch_size, 150, 3, 224, 224)
    audio = torch.randn(batch_size, 44100 * 5)
    sensors = torch.randn(batch_size, 15)

    print(f"\nInput shapes:")
    print(f"  Video: {video.shape}")
    print(f"  Audio: {audio.shape}")
    print(f"  Sensors: {sensors.shape}")

    # Testar variantes
    variants = [
        ('Lightweight', FusionVariants.lightweight()),
        ('Balanced', FusionVariants.balanced()),
        ('Accurate', FusionVariants.accurate()),
        ('Vision Focused', FusionVariants.vision_focused()),
        ('Audio Focused', FusionVariants.audio_focused()),
    ]

    print("\n" + "=" * 70)

    for name, model in variants:
        model.eval()

        with torch.no_grad():
            logits = model(video, audio, sensors)
            predicted_classes, confidences = model.predict(video, audio, sensors)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{name}:")
        print(f"  Output: {logits.shape}")
        print(f"  Predictions: {predicted_classes}")
        print(f"  Confidences: {confidences}")
        print(f"  Total params: {total_params:,}")
        print(f"  Trainable params: {trainable_params:,}")
        print(f"  Fusion strategy: {model.fusion_strategy}")

    # Benchmark de velocidade
    print("\n" + "=" * 70)
    print("BENCHMARK DE VELOCIDADE (Lightweight)")
    print("=" * 70)

    import time

    model = FusionVariants.lightweight()
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(video, audio, sensors)

    # Benchmark
    num_iterations = 20
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(video, audio, sensors)

    elapsed_time = time.time() - start_time
    avg_time = (elapsed_time / num_iterations) * 1000

    print(f"\nAverage inference time: {avg_time:.2f} ms")
    print(f"Throughput: {num_iterations / elapsed_time:.1f} samples/sec")
