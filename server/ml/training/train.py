"""
Script de Treinamento Principal

Treina modelo de fusão multimodal para detecção de anomalias
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional
import json
from tqdm import tqdm
import wandb
from dataclasses import dataclass, asdict

from ..models.video_encoder import VideoEncoder
from ..models.audio_encoder import AudioEncoder
from ..models.sensor_encoder import SensorEncoder
from ..models.fusion_classifier import FusionClassifier
from .dataset_loader import create_dataloaders


# ============================================================================
# CONFIGURAÇÃO
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuração de treinamento"""

    # Dataset
    dataset_root: str = "./dataset"
    batch_size: int = 8
    num_workers: int = 4
    train_split: float = 0.8

    # Model
    video_backbone: str = "efficientnet_b0"
    audio_architecture: str = "cnn_small"
    sensor_architecture: str = "simple"
    fusion_strategy: str = "concat"

    # Training
    epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 5
    use_mixed_precision: bool = True

    # Optimization
    optimizer: str = "adamw"  # adamw, sgd
    scheduler: str = "cosine"  # cosine, step

    # Regularization
    dropout: float = 0.3
    label_smoothing: float = 0.1

    # Checkpoints
    checkpoint_dir: str = "./checkpoints"
    save_every: int = 5

    # Logging
    use_wandb: bool = False
    wandb_project: str = "spectral"
    log_every: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    """Classe de treinamento"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Criar diretório de checkpoints
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Inicializar W&B
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                config=asdict(config)
            )

        # Criar modelo
        self.model = self._create_model()

        # Criar otimizador
        self.optimizer = self._create_optimizer()

        # Criar scheduler
        self.scheduler = self._create_scheduler()

        # Loss function
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing
        )

        # Mixed precision
        self.scaler = GradScaler() if config.use_mixed_precision else None

        # Métricas
        self.best_val_acc = 0.0
        self.current_epoch = 0

        print(f"✅ Trainer inicializado em {self.device}")

    def _create_model(self) -> nn.Module:
        """Cria modelo"""

        # Encoders
        video_encoder = VideoEncoder(
            backbone=self.config.video_backbone,
            pretrained=True,
            dropout=self.config.dropout
        )

        audio_encoder = AudioEncoder(
            architecture=self.config.audio_architecture,
            dropout=self.config.dropout
        )

        sensor_encoder = SensorEncoder(
            architecture=self.config.sensor_architecture,
            dropout=self.config.dropout
        )

        # Fusion classifier
        model = FusionClassifier(
            video_encoder=video_encoder,
            audio_encoder=audio_encoder,
            sensor_encoder=sensor_encoder,
            num_classes=4,  # Normal, Anomaly, Interference, EVP
            fusion_strategy=self.config.fusion_strategy,
            dropout=self.config.dropout
        )

        model = model.to(self.device)

        # Contar parâmetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"   Parâmetros totais: {total_params:,}")
        print(f"   Parâmetros treináveis: {trainable_params:,}")

        return model

    def _create_optimizer(self) -> optim.Optimizer:
        """Cria otimizador"""

        if self.config.optimizer == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Otimizador desconhecido: {self.config.optimizer}")

        return optimizer

    def _create_scheduler(self):
        """Cria scheduler"""

        if self.config.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs,
                eta_min=1e-6
            )
        elif self.config.scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            scheduler = None

        return scheduler

    def train(
        self,
        train_loader,
        val_loader
    ):
        """Loop de treinamento principal"""

        print(f"\n🚀 Iniciando treinamento por {self.config.epochs} épocas\n")

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self._train_epoch(train_loader)

            # Validate
            val_metrics = self._validate_epoch(val_loader)

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Log
            self._log_epoch(train_metrics, val_metrics)

            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self._save_checkpoint(val_metrics['accuracy'])

            # Early stopping se val_acc melhorou
            if val_metrics['accuracy'] > self.best_val_acc:
                self.best_val_acc = val_metrics['accuracy']
                self._save_checkpoint(val_metrics['accuracy'], is_best=True)

        print(f"\n✅ Treinamento completo! Melhor val_acc: {self.best_val_acc:.2%}")

    def _train_epoch(self, train_loader) -> Dict:
        """Treina uma época"""

        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Época {self.current_epoch + 1} [Treino]")

        for batch_idx, batch in enumerate(pbar):
            # Mover para device
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            sensors = batch['sensors'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward
            self.optimizer.zero_grad()

            if self.scaler:
                with autocast():
                    outputs = self.model(video, audio, sensors)
                    loss = self.criterion(outputs, labels)

                # Backward
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(video, audio, sensors)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            # Métricas
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100.0 * correct / total
            })

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total
        }

    def _validate_epoch(self, val_loader) -> Dict:
        """Valida uma época"""

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # Confusion matrix
        num_classes = 4
        confusion = torch.zeros(num_classes, num_classes)

        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Época {self.current_epoch + 1} [Val]")

            for batch_idx, batch in enumerate(pbar):
                video = batch['video'].to(self.device)
                audio = batch['audio'].to(self.device)
                sensors = batch['sensors'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward
                outputs = self.model(video, audio, sensors)
                loss = self.criterion(outputs, labels)

                # Métricas
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update confusion matrix
                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion[t.long(), p.long()] += 1

                pbar.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': 100.0 * correct / total
                })

        # Calcular métricas por classe
        per_class_acc = {}
        for i in range(num_classes):
            class_correct = confusion[i, i].item()
            class_total = confusion[i].sum().item()
            if class_total > 0:
                per_class_acc[f'class_{i}_acc'] = class_correct / class_total

        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total,
            'confusion_matrix': confusion.tolist(),
            **per_class_acc
        }

    def _log_epoch(self, train_metrics: Dict, val_metrics: Dict):
        """Log de métricas"""

        print(f"\nÉpoca {self.current_epoch + 1}/{self.config.epochs}")
        print(f"  Treino - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2%}")
        print(f"  Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2%}")

        if self.config.use_wandb:
            wandb.log({
                'epoch': self.current_epoch,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })

    def _save_checkpoint(self, val_acc: float, is_best: bool = False):
        """Salva checkpoint"""

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'config': asdict(self.config)
        }

        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
            print(f"💾 Salvando melhor modelo: {path}")
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch + 1}.pth'

        torch.save(checkpoint, path)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Função principal de treinamento"""

    # Configuração
    config = TrainingConfig(
        dataset_root="./dataset",
        batch_size=8,
        epochs=100,
        learning_rate=1e-4,
        video_backbone="efficientnet_b0",
        fusion_strategy="attention",
        use_wandb=False
    )

    # Criar dataloaders
    train_loader, val_loader = create_dataloaders(
        Path(config.dataset_root),
        batch_size=config.batch_size,
        train_split=config.train_split,
        num_workers=config.num_workers
    )

    # Criar trainer
    trainer = Trainer(config)

    # Treinar
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
