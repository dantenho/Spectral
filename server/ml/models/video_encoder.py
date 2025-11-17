"""
Video Encoder - Múltiplas Arquiteturas

Implementa diferentes backbones para extração de features de vídeo
"""

import torch
import torch.nn as nn
import timm
from typing import Literal, Optional


class VideoEncoder(nn.Module):
    """
    Encoder de vídeo base com backbone configurável

    Suporta múltiplas arquiteturas pré-treinadas
    """

    def __init__(
        self,
        backbone: str = 'efficientnet_b0',
        pretrained: bool = True,
        freeze_backbone: bool = True,
        num_frames: int = 150,
        embedding_dim: Optional[int] = None,
        temporal_aggregation: Literal['mean', 'max', 'lstm', 'attention'] = 'mean'
    ):
        """
        Args:
            backbone: Nome do modelo do timm
            pretrained: Usar pesos pré-treinados
            freeze_backbone: Congelar backbone durante treinamento
            num_frames: Número de frames esperados
            embedding_dim: Dimensão final do embedding (None = usar dimensão do backbone)
            temporal_aggregation: Método de agregação temporal
        """
        super().__init__()

        self.num_frames = num_frames
        self.temporal_aggregation = temporal_aggregation

        # Criar backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''   # Remove global pooling
        )

        # Obter dimensão de saída do backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            dummy_output = self.backbone(dummy_input)

            if len(dummy_output.shape) == 4:  # [B, C, H, W]
                self.backbone_dim = dummy_output.shape[1]
                self.spatial_size = dummy_output.shape[2:]
                self.needs_pooling = True
            else:  # [B, C]
                self.backbone_dim = dummy_output.shape[1]
                self.needs_pooling = False

        # Freeze backbone se solicitado
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Agregação temporal
        if temporal_aggregation == 'lstm':
            self.temporal_lstm = nn.LSTM(
                input_size=self.backbone_dim,
                hidden_size=self.backbone_dim,
                num_layers=2,
                batch_first=True,
                dropout=0.3
            )
        elif temporal_aggregation == 'attention':
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=self.backbone_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )

        # Projection head (opcional)
        if embedding_dim and embedding_dim != self.backbone_dim:
            self.projection = nn.Sequential(
                nn.Linear(self.backbone_dim, embedding_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            self.final_dim = embedding_dim
        else:
            self.projection = None
            self.final_dim = self.backbone_dim

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_frames: [batch_size, num_frames, 3, H, W]

        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        batch_size, num_frames, C, H, W = video_frames.shape

        # Reshape para processar todos os frames de uma vez
        frames = video_frames.view(batch_size * num_frames, C, H, W)

        # Extrair features com backbone
        frame_features = self.backbone(frames)

        # Se precisar de pooling espacial
        if self.needs_pooling:
            frame_features = torch.mean(frame_features, dim=(2, 3))  # Global Average Pooling

        # Reshape de volta: [batch, num_frames, features]
        frame_features = frame_features.view(batch_size, num_frames, -1)

        # Agregação temporal
        if self.temporal_aggregation == 'mean':
            video_embedding = torch.mean(frame_features, dim=1)

        elif self.temporal_aggregation == 'max':
            video_embedding = torch.max(frame_features, dim=1)[0]

        elif self.temporal_aggregation == 'lstm':
            lstm_out, _ = self.temporal_lstm(frame_features)
            video_embedding = lstm_out[:, -1, :]  # Última saída

        elif self.temporal_aggregation == 'attention':
            attn_out, _ = self.temporal_attention(
                frame_features, frame_features, frame_features
            )
            video_embedding = torch.mean(attn_out, dim=1)

        # Projection (se configurado)
        if self.projection:
            video_embedding = self.projection(video_embedding)

        return video_embedding


# ============================================================================
# VARIANTES DE VIDEO ENCODER
# ============================================================================

class VideoEncoderVariants:
    """
    Factory para criar diferentes variantes de Video Encoder
    """

    @staticmethod
    def efficientnet_b0(pretrained: bool = True, **kwargs) -> VideoEncoder:
        """EfficientNet-B0: Rápido e eficiente"""
        return VideoEncoder(
            backbone='efficientnet_b0',
            pretrained=pretrained,
            **kwargs
        )

    @staticmethod
    def efficientnet_b2(pretrained: bool = True, **kwargs) -> VideoEncoder:
        """EfficientNet-B2: Melhor accuracy"""
        return VideoEncoder(
            backbone='efficientnet_b2',
            pretrained=pretrained,
            **kwargs
        )

    @staticmethod
    def resnet50(pretrained: bool = True, **kwargs) -> VideoEncoder:
        """ResNet50: Clássico e robusto"""
        return VideoEncoder(
            backbone='resnet50',
            pretrained=pretrained,
            **kwargs
        )

    @staticmethod
    def mobilenetv3_large(pretrained: bool = True, **kwargs) -> VideoEncoder:
        """MobileNetV3: Ultra rápido para edge"""
        return VideoEncoder(
            backbone='mobilenetv3_large_100',
            pretrained=pretrained,
            **kwargs
        )

    @staticmethod
    def convnext_tiny(pretrained: bool = True, **kwargs) -> VideoEncoder:
        """ConvNeXt: Estado da arte"""
        return VideoEncoder(
            backbone='convnext_tiny',
            pretrained=pretrained,
            **kwargs
        )

    @staticmethod
    def vit_small(pretrained: bool = True, **kwargs) -> VideoEncoder:
        """Vision Transformer: Atenção global"""
        return VideoEncoder(
            backbone='vit_small_patch16_224',
            pretrained=pretrained,
            freeze_backbone=False,  # ViT geralmente precisa fine-tuning
            **kwargs
        )


# ============================================================================
# TESTE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DE VIDEO ENCODERS")
    print("=" * 60)

    # Criar video de teste
    batch_size = 2
    num_frames = 150
    video = torch.randn(batch_size, num_frames, 3, 224, 224)

    print(f"\nInput shape: {video.shape}")

    # Testar diferentes variantes
    variants = [
        ('EfficientNet-B0', VideoEncoderVariants.efficientnet_b0()),
        ('ResNet50', VideoEncoderVariants.resnet50()),
        ('MobileNetV3', VideoEncoderVariants.mobilenetv3_large()),
    ]

    for name, encoder in variants:
        encoder.eval()
        with torch.no_grad():
            embedding = encoder(video)
        print(f"\n{name}:")
        print(f"  Output shape: {embedding.shape}")
        print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        print(f"  Trainable: {sum(p.numel() for p in encoder.parameters() if p.requires_grad):,}")

    # Testar agregação temporal
    print("\n" + "=" * 60)
    print("TESTE DE AGREGAÇÃO TEMPORAL")
    print("=" * 60)

    for agg_method in ['mean', 'max', 'lstm', 'attention']:
        encoder = VideoEncoder(
            backbone='efficientnet_b0',
            temporal_aggregation=agg_method
        )
        encoder.eval()

        with torch.no_grad():
            embedding = encoder(video)

        print(f"\n{agg_method.upper()}:")
        print(f"  Output: {embedding.shape}")
        print(f"  Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")
