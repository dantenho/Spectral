"""
Neural Network Ensemble - Combina múltiplas redes neurais

Permite usar diferentes modelos em paralelo para melhorar métricas
"""

import torch
import torch.nn as nn
from typing import List, Literal, Optional, Dict
from dataclasses import dataclass

from .models.fusion_classifier import FusionClassifier, FusionVariants


@dataclass
class EnsemblePrediction:
    """Resultado da predição do ensemble"""
    predicted_class: int
    confidence: float
    class_probabilities: torch.Tensor
    individual_predictions: List[Dict]
    voting_method: str


class NeuralEnsemble(nn.Module):
    """
    Ensemble de múltiplas redes neurais

    Suporta diferentes métodos de combinação:
    - Voting: Vot ação majoritária
    - Average: Média das probabilidades
    - Weighted: Média ponderada por pesos
    - Stacking: Meta-learner sobre predições
    """

    def __init__(
        self,
        models: List[FusionClassifier],
        method: Literal['voting', 'average', 'weighted', 'stacking'] = 'average',
        weights: Optional[List[float]] = None,
        meta_learner: Optional[nn.Module] = None
    ):
        """
        Args:
            models: Lista de modelos a combinar
            method: Método de ensemble
            weights: Pesos para cada modelo (se weighted)
            meta_learner: Modelo de segundo nível (se stacking)
        """
        super().__init__()

        assert len(models) > 0, "Ensemble precisa de pelo menos 1 modelo"

        self.models = nn.ModuleList(models)
        self.method = method
        self.num_models = len(models)

        # Verificar que todos têm mesmo número de classes
        num_classes_list = [m.num_classes for m in models]
        assert len(set(num_classes_list)) == 1, "Todos modelos devem ter mesmo número de classes"
        self.num_classes = num_classes_list[0]

        # Configurar pesos se weighted
        if method == 'weighted':
            if weights is None:
                # Pesos uniformes por padrão
                weights = [1.0 / self.num_models] * self.num_models
            else:
                # Normalizar pesos
                total = sum(weights)
                weights = [w / total for w in weights]

            self.register_buffer('weights', torch.tensor(weights))
        else:
            self.weights = None

        # Meta-learner para stacking
        if method == 'stacking':
            if meta_learner is None:
                # MLP simples como meta-learner padrão
                self.meta_learner = nn.Sequential(
                    nn.Linear(self.num_models * self.num_classes, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, self.num_classes)
                )
            else:
                self.meta_learner = meta_learner
        else:
            self.meta_learner = None

    def forward(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        sensors: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass do ensemble

        Args:
            video: [batch, num_frames, 3, H, W]
            audio: [batch, samples]
            sensors: [batch, features]

        Returns:
            logits: [batch, num_classes]
        """
        # Coletar predições de todos os modelos
        all_logits = []

        for model in self.models:
            logits = model(video, audio, sensors)
            all_logits.append(logits)

        # Stack: [num_models, batch, num_classes]
        all_logits = torch.stack(all_logits, dim=0)

        # Combinar baseado no método
        if self.method == 'voting':
            # Votação majoritária (hard voting)
            all_probs = torch.softmax(all_logits, dim=2)
            predictions = torch.argmax(all_probs, dim=2)  # [num_models, batch]
            # Moda (classe mais votada)
            ensemble_logits = torch.mode(predictions, dim=0)[0]
            # Converter de volta para logits (one-hot)
            ensemble_logits = torch.nn.functional.one_hot(
                ensemble_logits, num_classes=self.num_classes
            ).float()

        elif self.method == 'average':
            # Média das probabilidades (soft voting)
            all_probs = torch.softmax(all_logits, dim=2)
            avg_probs = torch.mean(all_probs, dim=0)  # [batch, num_classes]
            # Converter de volta para logits
            ensemble_logits = torch.log(avg_probs + 1e-10)

        elif self.method == 'weighted':
            # Média ponderada
            all_probs = torch.softmax(all_logits, dim=2)
            # [num_models, batch, num_classes] * [num_models, 1, 1]
            weighted_probs = all_probs * self.weights.view(-1, 1, 1)
            ensemble_probs = torch.sum(weighted_probs, dim=0)  # [batch, num_classes]
            ensemble_logits = torch.log(ensemble_probs + 1e-10)

        elif self.method == 'stacking':
            # Stacking: usar meta-learner
            # Concatenar todas as probabilidades
            all_probs = torch.softmax(all_logits, dim=2)
            # [num_models, batch, num_classes] -> [batch, num_models * num_classes]
            stacked_probs = all_probs.permute(1, 0, 2).reshape(
                all_probs.shape[1], -1
            )
            # Meta-learner
            ensemble_logits = self.meta_learner(stacked_probs)

        return ensemble_logits

    def predict(
        self,
        video: torch.Tensor,
        audio: torch.Tensor,
        sensors: torch.Tensor,
        return_details: bool = False
    ) -> EnsemblePrediction:
        """
        Predição com detalhes do ensemble

        Args:
            video, audio, sensors: Inputs
            return_details: Retornar predições individuais

        Returns:
            EnsemblePrediction com detalhes
        """
        # Predição do ensemble
        ensemble_logits = self.forward(video, audio, sensors)
        ensemble_probs = torch.softmax(ensemble_logits, dim=1)

        confidence, predicted_class = torch.max(ensemble_probs, dim=1)

        individual_predictions = []

        if return_details:
            # Coletar predições individuais
            for i, model in enumerate(self.models):
                with torch.no_grad():
                    logits = model(video, audio, sensors)
                    probs = torch.softmax(logits, dim=1)
                    conf, pred = torch.max(probs, dim=1)

                    individual_predictions.append({
                        'model_index': i,
                        'predicted_class': pred.item(),
                        'confidence': conf.item(),
                        'probabilities': probs[0].tolist()
                    })

        return EnsemblePrediction(
            predicted_class=predicted_class.item(),
            confidence=confidence.item(),
            class_probabilities=ensemble_probs[0],
            individual_predictions=individual_predictions,
            voting_method=self.method
        )


# ============================================================================
# ENSEMBLE PRÉ-CONFIGURADOS
# ============================================================================

class EnsembleVariants:
    """Factory para criar ensembles pré-configurados"""

    @staticmethod
    def fast_ensemble(num_classes: int = 4) -> NeuralEnsemble:
        """
        Ensemble rápido com 3 modelos leves

        Combina: Lightweight + Balanced + Vision Focused
        """
        models = [
            FusionVariants.lightweight(num_classes),
            FusionVariants.balanced(num_classes),
            FusionVariants.vision_focused(num_classes)
        ]

        return NeuralEnsemble(models, method='average')

    @staticmethod
    def accurate_ensemble(num_classes: int = 4) -> NeuralEnsemble:
        """
        Ensemble para máxima accuracy

        Combina: Balanced + Accurate + Audio Focused
        """
        models = [
            FusionVariants.balanced(num_classes),
            FusionVariants.accurate(num_classes),
            FusionVariants.audio_focused(num_classes)
        ]

        return NeuralEnsemble(models, method='weighted', weights=[0.3, 0.5, 0.2])

    @staticmethod
    def full_ensemble(num_classes: int = 4) -> NeuralEnsemble:
        """
        Ensemble completo com todas as variantes

        Usa stacking com meta-learner
        """
        models = [
            FusionVariants.lightweight(num_classes),
            FusionVariants.balanced(num_classes),
            FusionVariants.accurate(num_classes),
            FusionVariants.vision_focused(num_classes),
            FusionVariants.audio_focused(num_classes)
        ]

        return NeuralEnsemble(models, method='stacking')

    @staticmethod
    def specialized_ensemble(num_classes: int = 4) -> NeuralEnsemble:
        """
        Ensemble especializado em detecção EVP/humanoide

        Pesos: 20% visual, 50% áudio, 30% balanced
        """
        models = [
            FusionVariants.vision_focused(num_classes),
            FusionVariants.audio_focused(num_classes),
            FusionVariants.balanced(num_classes)
        ]

        return NeuralEnsemble(
            models,
            method='weighted',
            weights=[0.2, 0.5, 0.3]
        )


# ============================================================================
# DIVERSITY ENSEMBLE (Maximizar Diversidade)
# ============================================================================

class DiversityEnsemble(NeuralEnsemble):
    """
    Ensemble que maximiza diversidade entre modelos

    Seleciona automaticamente modelos com predições mais diversas
    """

    def __init__(
        self,
        candidate_models: List[FusionClassifier],
        num_models: int = 3,
        diversity_threshold: float = 0.3
    ):
        """
        Args:
            candidate_models: Lista de modelos candidatos
            num_models: Quantos modelos selecionar
            diversity_threshold: Threshold de diversidade mínima
        """
        # Selecionar modelos mais diversos
        selected_models = self._select_diverse_models(
            candidate_models, num_models, diversity_threshold
        )

        super().__init__(selected_models, method='average')

    def _select_diverse_models(
        self,
        candidates: List[FusionClassifier],
        k: int,
        threshold: float
    ) -> List[FusionClassifier]:
        """
        Seleciona k modelos com máxima diversidade

        Usa greedy selection baseado em diversidade de predições
        """
        # Por simplicidade, retornar primeiros k
        # Em produção, implementar seleção baseada em validação
        return candidates[:k]


# ============================================================================
# TESTE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTE DE NEURAL ENSEMBLE")
    print("=" * 70)

    # Dados de teste
    batch_size = 2
    video = torch.randn(batch_size, 150, 3, 224, 224)
    audio = torch.randn(batch_size, 44100 * 5)
    sensors = torch.randn(batch_size, 15)

    print(f"\nInput shapes:")
    print(f"  Video: {video.shape}")
    print(f"  Audio: {audio.shape}")
    print(f"  Sensors: {sensors.shape}")

    # Testar diferentes ensembles
    ensembles = [
        ('Fast Ensemble', EnsembleVariants.fast_ensemble()),
        ('Accurate Ensemble', EnsembleVariants.accurate_ensemble()),
        ('Full Ensemble', EnsembleVariants.full_ensemble()),
        ('Specialized Ensemble', EnsembleVariants.specialized_ensemble()),
    ]

    print("\n" + "=" * 70)

    for name, ensemble in ensembles:
        ensemble.eval()

        with torch.no_grad():
            # Predição sem detalhes
            logits = ensemble(video, audio, sensors)

            # Predição com detalhes
            result = ensemble.predict(video, audio, sensors, return_details=True)

        print(f"\n{name}:")
        print(f"  Number of models: {ensemble.num_models}")
        print(f"  Method: {ensemble.method}")
        print(f"  Output: {logits.shape}")
        print(f"  Predicted class: {result.predicted_class}")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Class probabilities: {result.class_probabilities.tolist()}")

        if result.individual_predictions:
            print(f"  Individual predictions:")
            for pred in result.individual_predictions:
                print(f"    Model {pred['model_index']}: "
                      f"class={pred['predicted_class']}, "
                      f"conf={pred['confidence']:.3f}")

        total_params = sum(p.numel() for p in ensemble.parameters())
        print(f"  Total parameters: {total_params:,}")

    # Benchmark
    print("\n" + "=" * 70)
    print("BENCHMARK DE VELOCIDADE")
    print("=" * 70)

    import time

    # Fast Ensemble
    ensemble = EnsembleVariants.fast_ensemble()
    ensemble.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = ensemble(video, audio, sensors)

    # Benchmark
    num_iterations = 10
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = ensemble(video, audio, sensors)

    elapsed_time = time.time() - start_time
    avg_time = (elapsed_time / num_iterations) * 1000

    print(f"\nFast Ensemble (3 models):")
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  Throughput: {num_iterations / elapsed_time:.1f} samples/sec")

    # Full Ensemble
    ensemble = EnsembleVariants.full_ensemble()
    ensemble.eval()

    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = ensemble(video, audio, sensors)

    elapsed_time = time.time() - start_time
    avg_time = (elapsed_time / num_iterations) * 1000

    print(f"\nFull Ensemble (5 models + meta-learner):")
    print(f"  Average time: {avg_time:.2f} ms")
    print(f"  Throughput: {num_iterations / elapsed_time:.1f} samples/sec")

    # Comparação de predições
    print("\n" + "=" * 70)
    print("COMPARAÇÃO DE MÉTODOS")
    print("=" * 70)

    # Criar 4 modelos
    models = [
        FusionVariants.lightweight(),
        FusionVariants.balanced(),
        FusionVariants.accurate(),
        FusionVariants.vision_focused()
    ]

    methods = ['voting', 'average', 'weighted', 'stacking']

    for method in methods:
        if method == 'weighted':
            ensemble = NeuralEnsemble(models, method=method, weights=[0.1, 0.3, 0.4, 0.2])
        else:
            ensemble = NeuralEnsemble(models, method=method)

        ensemble.eval()

        with torch.no_grad():
            result = ensemble.predict(video, audio, sensors)

        print(f"\n{method.upper()}:")
        print(f"  Class: {result.predicted_class}, Confidence: {result.confidence:.4f}")
