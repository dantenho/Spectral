"""
Quantização de Modelos - INT4 e INT8

Converte modelos PyTorch para versões quantizadas para mobile/edge deployment
"""

import torch
import torch.nn as nn
from torch.quantization import (
    quantize_dynamic,
    quantize_qat,
    get_default_qconfig,
    prepare_qat,
    convert
)
from pathlib import Path
from typing import Optional, Literal
import json

# TensorFlow Lite para quantização adicional
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow não disponível. Quantização TFLite desabilitada.")


# ============================================================================
# PYTORCH QUANTIZATION
# ============================================================================

class ModelQuantizer:
    """
    Quantizador de modelos PyTorch

    Suporta:
    - Dynamic quantization (INT8)
    - Post-training quantization (INT8)
    - Quantization-aware training (INT8)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()

    def dynamic_quantize_int8(self) -> nn.Module:
        """
        Dynamic quantization para INT8

        Quantiza pesos para INT8, ativações permanecem FP32
        Melhor para modelos com muitas operações lineares
        """

        print("🔧 Aplicando Dynamic Quantization INT8...")

        quantized_model = quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d, nn.LSTM, nn.GRU},
            dtype=torch.qint8
        )

        print("✅ Dynamic quantization completa")

        return quantized_model

    def static_quantize_int8(
        self,
        calibration_loader,
        num_calibration_batches: int = 100
    ) -> nn.Module:
        """
        Post-training static quantization para INT8

        Quantiza pesos e ativações para INT8
        Requer dados de calibração
        """

        print("🔧 Aplicando Static Quantization INT8...")

        # Preparar modelo para quantização
        self.model.qconfig = get_default_qconfig('x86')  # ou 'qnnpack' para ARM

        # Fuse modules (Conv+BN+ReLU)
        torch.quantization.fuse_modules(
            self.model,
            [['conv', 'bn', 'relu']],
            inplace=True
        )

        # Prepare
        prepared_model = torch.quantization.prepare(self.model)

        # Calibração
        prepared_model.eval()

        print(f"   Calibrando com {num_calibration_batches} batches...")

        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= num_calibration_batches:
                    break

                # Forward pass
                video = batch['video']
                audio = batch['audio']
                sensors = batch['sensors']

                try:
                    _ = prepared_model(video, audio, sensors)
                except Exception as e:
                    print(f"   ⚠️  Erro na calibração batch {i}: {e}")
                    continue

        # Convert
        quantized_model = torch.quantization.convert(prepared_model)

        print("✅ Static quantization completa")

        return quantized_model

    def save_quantized(
        self,
        quantized_model: nn.Module,
        output_path: Path,
        metadata: Optional[dict] = None
    ):
        """Salva modelo quantizado"""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Salvar modelo
        torch.jit.save(
            torch.jit.script(quantized_model),
            str(output_path)
        )

        # Salvar metadata
        if metadata:
            metadata_path = output_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

        print(f"💾 Modelo quantizado salvo em {output_path}")

        # Tamanho do arquivo
        size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"   Tamanho: {size_mb:.2f} MB")


# ============================================================================
# TFLITE QUANTIZATION (INT4 via INT8)
# ============================================================================

class TFLiteQuantizer:
    """
    Quantizador para TensorFlow Lite

    Nota: TFLite não tem suporte nativo para INT4, mas podemos:
    1. Usar INT8 quantization (muito próximo)
    2. Usar float16 (meio caminho)
    3. Implementar INT4 custom (avançado)
    """

    def __init__(self, model_path: Path):
        if not TF_AVAILABLE:
            raise RuntimeError("TensorFlow não disponível")

        self.model_path = model_path

    def convert_to_tflite_int8(
        self,
        output_path: Path,
        representative_dataset_gen,
        int_only: bool = True
    ):
        """
        Converte modelo para TFLite com quantização INT8

        Args:
            output_path: Caminho de saída
            representative_dataset_gen: Gerador de dados representativos
            int_only: Se True, força INT8 para tudo (incluindo inputs/outputs)
        """

        print("🔧 Convertendo para TFLite INT8...")

        # Carregar modelo
        converter = tf.lite.TFLiteConverter.from_saved_model(str(self.model_path))

        # Configurar quantização
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        if int_only:
            # INT8 para tudo
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        # Representative dataset para calibração
        converter.representative_dataset = representative_dataset_gen

        # Converter
        tflite_model = converter.convert()

        # Salvar
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"✅ TFLite INT8 criado: {output_path} ({size_mb:.2f} MB)")

    def convert_to_tflite_float16(
        self,
        output_path: Path
    ):
        """
        Converte modelo para TFLite com quantização FLOAT16

        Reduz tamanho em ~50% mantendo boa precisão
        """

        print("🔧 Convertendo para TFLite FLOAT16...")

        converter = tf.lite.TFLiteConverter.from_saved_model(str(self.model_path))

        # FLOAT16
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

        # Converter
        tflite_model = converter.convert()

        # Salvar
        output_path = Path(output_path)
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

        size_mb = output_path.stat().st_size / (1024 ** 2)
        print(f"✅ TFLite FLOAT16 criado: {output_path} ({size_mb:.2f} MB)")


# ============================================================================
# INT4 PSEUDO-QUANTIZATION
# ============================================================================

class INT4PseudoQuantizer:
    """
    Pseudo-quantização INT4

    Como TFLite não suporta INT4 nativamente, implementamos:
    - Quantização de pesos para 4 bits
    - Packing de 2 valores INT4 em 1 INT8
    - Dequantização em runtime

    Isso reduz tamanho do modelo mas mantém operações em INT8/FP16
    """

    @staticmethod
    def quantize_weights_int4(weights: torch.Tensor) -> tuple[torch.Tensor, float, float]:
        """
        Quantiza pesos para INT4 (-8 a 7)

        Returns:
            (quantized_weights, scale, zero_point)
        """

        # Calcular min/max
        w_min = weights.min()
        w_max = weights.max()

        # Calcular scale e zero_point para INT4 (-8 a 7)
        qmin = -8
        qmax = 7
        scale = (w_max - w_min) / (qmax - qmin)
        zero_point = qmin - w_min / scale

        # Quantizar
        quantized = torch.round(weights / scale + zero_point)
        quantized = torch.clamp(quantized, qmin, qmax).to(torch.int8)

        return quantized, scale.item(), zero_point.item()

    @staticmethod
    def dequantize_weights_int4(
        quantized: torch.Tensor,
        scale: float,
        zero_point: float
    ) -> torch.Tensor:
        """Dequantiza pesos INT4"""

        return (quantized.float() - zero_point) * scale

    @staticmethod
    def pack_int4_to_int8(values: torch.Tensor) -> torch.Tensor:
        """
        Empacota 2 valores INT4 em 1 valor INT8

        Exemplo:
            [3, 5] -> (3 << 4) | (5 & 0x0F) = 0x35 = 53
        """

        # Garantir número par de elementos
        if values.numel() % 2 != 0:
            values = torch.cat([values, torch.zeros(1, dtype=torch.int8)])

        # Reshape para pares
        values = values.view(-1, 2)

        # Pack: high nibble = values[:, 0], low nibble = values[:, 1]
        packed = (values[:, 0] << 4) | (values[:, 1] & 0x0F)

        return packed.to(torch.int8)


# ============================================================================
# COMPARAÇÃO DE MODELOS
# ============================================================================

def compare_models(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_loader,
    num_samples: int = 100
):
    """
    Compara precisão entre modelo original e quantizado

    Returns:
        Dict com métricas de comparação
    """

    print("\n📊 Comparando modelos...")

    original_model.eval()
    quantized_model.eval()

    original_correct = 0
    quantized_correct = 0
    total = 0

    mse_outputs = 0.0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break

            video = batch['video']
            audio = batch['audio']
            sensors = batch['sensors']
            labels = batch['label']

            # Original
            outputs_orig = original_model(video, audio, sensors)
            _, pred_orig = outputs_orig.max(1)
            original_correct += pred_orig.eq(labels).sum().item()

            # Quantized
            outputs_quant = quantized_model(video, audio, sensors)
            _, pred_quant = outputs_quant.max(1)
            quantized_correct += pred_quant.eq(labels).sum().item()

            # MSE
            mse_outputs += torch.mean((outputs_orig - outputs_quant) ** 2).item()

            total += labels.size(0)

    metrics = {
        'original_accuracy': original_correct / total,
        'quantized_accuracy': quantized_correct / total,
        'accuracy_drop': (original_correct - quantized_correct) / total,
        'mse_outputs': mse_outputs / num_samples
    }

    print(f"   Original Acc: {metrics['original_accuracy']:.2%}")
    print(f"   Quantized Acc: {metrics['quantized_accuracy']:.2%}")
    print(f"   Accuracy Drop: {metrics['accuracy_drop']:.2%}")
    print(f"   MSE Outputs: {metrics['mse_outputs']:.6f}")

    return metrics


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Exemplo de uso"""

    print("=" * 70)
    print("QUANTIZAÇÃO DE MODELOS - INT4 e INT8")
    print("=" * 70)

    # Carregar modelo treinado
    model_path = Path("./checkpoints/best_model.pth")

    if not model_path.exists():
        print(f"⚠️  Modelo não encontrado: {model_path}")
        print("   Treine um modelo primeiro com train.py")
        return

    # Carregar checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')

    # Recriar modelo (simplificado - adaptar conforme necessário)
    from ..models.fusion_classifier import create_fusion_variant

    model = create_fusion_variant('balanced')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Modelo carregado: {model_path}")

    # Quantizar
    quantizer = ModelQuantizer(model)

    # INT8 Dynamic
    quantized_int8 = quantizer.dynamic_quantize_int8()

    # Salvar
    output_dir = Path("./quantized_models")
    quantizer.save_quantized(
        quantized_int8,
        output_dir / "model_int8_dynamic.pt",
        metadata={'quantization': 'dynamic_int8', 'backend': 'pytorch'}
    )

    print("\n✅ Quantização completa!")


if __name__ == "__main__":
    main()
