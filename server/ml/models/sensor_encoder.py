"""
Sensor Encoder - Processa dados tabulares de sensores

MLP simples para extrair features de dados de sensores
"""

import torch
import torch.nn as nn
from typing import Optional


class SensorEncoder(nn.Module):
    """
    MLP para processar dados tabulares de sensores

    Input: Features estatísticas dos sensores (magnitude média, std, etc.)
    Output: Embedding de dimensão fixa
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dims: Optional[list[int]] = None,
        embedding_dim: int = 64,
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Dimensão de entrada (número de features)
            hidden_dims: Lista de dimensões das camadas ocultas
            embedding_dim: Dimensão do embedding final
            dropout: Taxa de dropout
            use_batch_norm: Usar Batch Normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # Hidden dimensions padrão se não especificado
        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Construir MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Camada final
        layers.append(nn.Linear(prev_dim, embedding_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, sensor_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_features: [batch_size, input_dim]

        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        return self.mlp(sensor_features)


class SensorEncoderDeep(nn.Module):
    """
    MLP profundo com skip connections (ResNet-style)

    Melhor para dados com muitas features
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 128,
        num_layers: int = 4,
        embedding_dim: int = 64,
        dropout: float = 0.3
    ):
        """
        Args:
            input_dim: Dimensão de entrada
            hidden_dim: Dimensão das camadas ocultas
            num_layers: Número de blocos residuais
            embedding_dim: Dimensão do embedding final
            dropout: Taxa de dropout
        """
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            self._make_residual_block(hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.embedding_dim = embedding_dim

    def _make_residual_block(self, hidden_dim: int, dropout: float) -> nn.Module:
        """Cria bloco residual"""
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim)
        )

    def forward(self, sensor_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_features: [batch_size, input_dim]

        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        x = self.input_proj(sensor_features)

        # Residual blocks
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual  # Skip connection
            x = torch.relu(x)

        x = self.output_proj(x)

        return x


class SensorEncoderAttention(nn.Module):
    """
    Encoder com self-attention para capturar relações entre features

    Útil quando features dos sensores têm interdependências
    """

    def __init__(
        self,
        input_dim: int = 15,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        embedding_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimensão de entrada
            hidden_dim: Dimensão das camadas ocultas
            num_heads: Número de heads de atenção
            num_layers: Número de camadas de atenção
            embedding_dim: Dimensão do embedding final
            dropout: Taxa de dropout
        """
        super().__init__()

        # Expandir features para sequência
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Positional encoding (já que tratamos features como sequência)
        self.pos_encoding = nn.Parameter(torch.randn(1, input_dim, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.embedding_dim = embedding_dim

    def forward(self, sensor_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sensor_features: [batch_size, input_dim]

        Returns:
            embeddings: [batch_size, embedding_dim]
        """
        batch_size, input_dim = sensor_features.shape

        # Expandir para sequência: cada feature é um token
        # [B, D] -> [B, D, 1] -> [B, D, hidden]
        x = sensor_features.unsqueeze(2)  # [B, D, 1]
        x = self.input_proj(x.transpose(1, 2))  # [B, 1, hidden]

        # Repetir para ter uma sequência
        x = x.expand(-1, input_dim, -1)  # [B, D, hidden]

        # Add positional encoding
        x = x + self.pos_encoding[:, :input_dim, :]

        # Transformer
        x = self.transformer(x)

        # Global pooling
        x = torch.mean(x, dim=1)  # [B, hidden]

        # Output projection
        x = self.output_proj(x)

        return x


# ============================================================================
# VARIANTES
# ============================================================================

class SensorEncoderVariants:
    """Factory para criar diferentes variantes"""

    @staticmethod
    def simple(input_dim: int = 15, embedding_dim: int = 64) -> SensorEncoder:
        """MLP simples e rápido"""
        return SensorEncoder(
            input_dim=input_dim,
            hidden_dims=[128, 64],
            embedding_dim=embedding_dim,
            dropout=0.2
        )

    @staticmethod
    def deep(input_dim: int = 15, embedding_dim: int = 64) -> SensorEncoderDeep:
        """MLP profundo com skip connections"""
        return SensorEncoderDeep(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=4,
            embedding_dim=embedding_dim,
            dropout=0.3
        )

    @staticmethod
    def attention(input_dim: int = 15, embedding_dim: int = 64) -> SensorEncoderAttention:
        """Encoder com self-attention"""
        return SensorEncoderAttention(
            input_dim=input_dim,
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            embedding_dim=embedding_dim,
            dropout=0.1
        )


# ============================================================================
# TESTE
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("TESTE DE SENSOR ENCODERS")
    print("=" * 60)

    # Criar dados de teste
    batch_size = 8
    input_dim = 15
    sensor_data = torch.randn(batch_size, input_dim)

    print(f"\nInput shape: {sensor_data.shape}")

    # Testar variantes
    variants = [
        ('Simple MLP', SensorEncoderVariants.simple()),
        ('Deep ResNet', SensorEncoderVariants.deep()),
        ('Attention', SensorEncoderVariants.attention()),
    ]

    for name, encoder in variants:
        encoder.eval()
        with torch.no_grad():
            embedding = encoder(sensor_data)

        print(f"\n{name}:")
        print(f"  Output shape: {embedding.shape}")
        print(f"  Parameters: {sum(p.numel() for p in encoder.parameters()):,}")
        print(f"  Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")

    # Benchmark
    print("\n" + "=" * 60)
    print("BENCHMARK DE VELOCIDADE")
    print("=" * 60)

    import time

    encoder = SensorEncoderVariants.simple()
    encoder.eval()

    num_iterations = 10000
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = encoder(sensor_data)

    elapsed_time = time.time() - start_time
    avg_time = (elapsed_time / num_iterations) * 1000000  # microsegundos

    print(f"\nSimple MLP:")
    print(f"  Avg time: {avg_time:.2f} µs")
    print(f"  Throughput: {num_iterations / elapsed_time:.0f} samples/sec")
