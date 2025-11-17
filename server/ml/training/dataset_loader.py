"""
Dataset Loader para Treinamento

Carrega e processa dados de vídeo, áudio e sensores para treinamento
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
import librosa

from ..models.video_encoder import VideoEncoder
from ..models.audio_encoder import AudioEncoder
from ..models.sensor_encoder import SensorEncoder


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AnomalySample:
    """Amostra de dados para treinamento"""
    video_path: Path
    audio_path: Path
    sensor_path: Path
    label: int  # 0=Normal, 1=Anomalia, 2=Interferência, 3=EVP
    metadata: Dict


# ============================================================================
# DATASET
# ============================================================================

class SpectralDataset(Dataset):
    """
    Dataset multimodal para detecção de anomalias

    Estrutura esperada do dataset:
    dataset/
        normal/
            video/
                sample_001.mp4
                sample_002.mp4
            audio/
                sample_001.wav
                sample_002.wav
            sensors/
                sample_001.json
                sample_002.json
        anomaly/
            ...
        interference/
            ...
        evp/
            ...
    """

    # Mapeamento de labels
    LABEL_MAP = {
        'normal': 0,
        'anomaly': 1,
        'interference': 2,
        'evp': 3
    }

    REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

    def __init__(
        self,
        dataset_root: Path,
        num_frames: int = 16,
        frame_size: Tuple[int, int] = (224, 224),
        audio_sample_rate: int = 44100,
        audio_duration: float = 3.0,
        n_mels: int = 128,
        sensor_sequence_length: int = 100,
        transform=None,
        augment: bool = False
    ):
        self.dataset_root = Path(dataset_root)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration = audio_duration
        self.n_mels = n_mels
        self.sensor_sequence_length = sensor_sequence_length
        self.transform = transform
        self.augment = augment

        # Carregar samples
        self.samples = self._load_samples()

        print(f"✅ Dataset carregado: {len(self.samples)} amostras")
        self._print_stats()

    def _load_samples(self) -> List[AnomalySample]:
        """Carrega lista de samples do dataset"""

        samples = []

        for category in ['normal', 'anomaly', 'interference', 'evp']:
            category_path = self.dataset_root / category

            if not category_path.exists():
                print(f"⚠️  Categoria {category} não encontrada")
                continue

            video_path = category_path / 'video'
            audio_path = category_path / 'audio'
            sensor_path = category_path / 'sensors'

            # Listar vídeos
            video_files = list(video_path.glob('*.mp4')) if video_path.exists() else []

            for video_file in video_files:
                # Nome base do arquivo (sem extensão)
                base_name = video_file.stem

                # Paths correspondentes
                audio_file = audio_path / f"{base_name}.wav"
                sensor_file = sensor_path / f"{base_name}.json"

                # Verificar se todos existem
                if not audio_file.exists():
                    print(f"⚠️  Áudio não encontrado para {base_name}")
                    continue

                if not sensor_file.exists():
                    print(f"⚠️  Sensores não encontrados para {base_name}")
                    continue

                # Criar sample
                sample = AnomalySample(
                    video_path=video_file,
                    audio_path=audio_file,
                    sensor_path=sensor_file,
                    label=self.LABEL_MAP[category],
                    metadata={'category': category, 'base_name': base_name}
                )

                samples.append(sample)

        return samples

    def _print_stats(self):
        """Imprime estatísticas do dataset"""

        label_counts = {}
        for sample in self.samples:
            label_counts[sample.label] = label_counts.get(sample.label, 0) + 1

        print("\nEstatísticas do Dataset:")
        for label, count in sorted(label_counts.items()):
            category = self.REVERSE_LABEL_MAP[label]
            print(f"  {category}: {count} amostras")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Retorna uma amostra do dataset"""

        sample = self.samples[idx]

        # 1. Carregar vídeo
        video_tensor = self._load_video(sample.video_path)

        # 2. Carregar áudio
        audio_tensor = self._load_audio(sample.audio_path)

        # 3. Carregar sensores
        sensor_tensor = self._load_sensors(sample.sensor_path)

        # 4. Label
        label = torch.tensor(sample.label, dtype=torch.long)

        return {
            'video': video_tensor,
            'audio': audio_tensor,
            'sensors': sensor_tensor,
            'label': label,
            'metadata': sample.metadata
        }

    def _load_video(self, video_path: Path) -> torch.Tensor:
        """
        Carrega vídeo e extrai frames

        Returns:
            Tensor de shape (num_frames, 3, H, W)
        """

        cap = cv2.VideoCapture(str(video_path))

        # Obter total de frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calcular índices de frames para amostrar uniformemente
        if total_frames >= self.num_frames:
            frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        else:
            # Se vídeo tem menos frames, repetir
            frame_indices = np.arange(total_frames)
            frame_indices = np.resize(frame_indices, self.num_frames)

        frames = []

        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if not ret:
                # Frame inválido, usar frame preto
                frame = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
            else:
                # Resize
                frame = cv2.resize(frame, self.frame_size)

                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

        cap.release()

        # Converter para tensor
        frames = np.array(frames)  # (num_frames, H, W, 3)
        frames = torch.from_numpy(frames).float() / 255.0  # Normalizar [0, 1]
        frames = frames.permute(0, 3, 1, 2)  # (num_frames, 3, H, W)

        # Normalizar para ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (frames - mean) / std

        return frames

    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """
        Carrega áudio e converte para Mel-spectrogram

        Returns:
            Tensor de shape (1, n_mels, time_steps)
        """

        # Carregar áudio
        audio, sr = librosa.load(str(audio_path), sr=self.audio_sample_rate)

        # Garantir duração fixa
        target_length = int(self.audio_sample_rate * self.audio_duration)

        if len(audio) > target_length:
            # Cortar
            audio = audio[:target_length]
        elif len(audio) < target_length:
            # Pad com zeros
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')

        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.audio_sample_rate,
            n_mels=self.n_mels,
            fmax=self.audio_sample_rate // 2
        )

        # Log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalizar [-1, 1]
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)

        # Converter para tensor
        mel_tensor = torch.from_numpy(mel_spec_db).float().unsqueeze(0)  # (1, n_mels, time)

        return mel_tensor

    def _load_sensors(self, sensor_path: Path) -> torch.Tensor:
        """
        Carrega dados de sensores

        Formato JSON esperado:
        {
            "samples": [
                {
                    "timestamp": float,
                    "accelerometer": {"x": float, "y": float, "z": float},
                    "gyroscope": {"x": float, "y": float, "z": float},
                    "magnetometer": {"x": float, "y": float, "z": float}
                },
                ...
            ]
        }

        Returns:
            Tensor de shape (sequence_length, 9)
                [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z]
        """

        with open(sensor_path, 'r') as f:
            data = json.load(f)

        samples = data.get('samples', [])

        # Extrair features
        features = []

        for sample in samples:
            accel = sample.get('accelerometer', {})
            gyro = sample.get('gyroscope', {})
            mag = sample.get('magnetometer', {})

            feature_vector = [
                accel.get('x', 0.0), accel.get('y', 0.0), accel.get('z', 0.0),
                gyro.get('x', 0.0), gyro.get('y', 0.0), gyro.get('z', 0.0),
                mag.get('x', 0.0), mag.get('y', 0.0), mag.get('z', 0.0)
            ]

            features.append(feature_vector)

        features = np.array(features)

        # Garantir tamanho fixo
        if len(features) > self.sensor_sequence_length:
            # Subsample uniformemente
            indices = np.linspace(0, len(features) - 1, self.sensor_sequence_length, dtype=int)
            features = features[indices]
        elif len(features) < self.sensor_sequence_length:
            # Pad com zeros
            pad_length = self.sensor_sequence_length - len(features)
            features = np.pad(features, ((0, pad_length), (0, 0)), mode='constant')

        # Normalizar (z-score)
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)

        # Converter para tensor
        sensor_tensor = torch.from_numpy(features).float()  # (seq_len, 9)

        return sensor_tensor


# ============================================================================
# DATA LOADER FACTORY
# ============================================================================

def create_dataloaders(
    dataset_root: Path,
    batch_size: int = 8,
    train_split: float = 0.8,
    num_workers: int = 4,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Cria DataLoaders de treino e validação

    Args:
        dataset_root: Raiz do dataset
        batch_size: Tamanho do batch
        train_split: Proporção para treino
        num_workers: Número de workers
        **dataset_kwargs: Argumentos para SpectralDataset

    Returns:
        (train_loader, val_loader)
    """

    # Criar dataset completo
    full_dataset = SpectralDataset(dataset_root, **dataset_kwargs)

    # Split train/val
    total_size = len(full_dataset)
    train_size = int(total_size * train_split)
    val_size = total_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Criar loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"\n✅ DataLoaders criados:")
    print(f"   Treino: {len(train_dataset)} amostras ({len(train_loader)} batches)")
    print(f"   Validação: {len(val_dataset)} amostras ({len(val_loader)} batches)")

    return train_loader, val_loader


# ============================================================================
# TESTE
# ============================================================================

if __name__ == "__main__":
    # Testar dataset
    dataset_root = Path("./dataset")

    if not dataset_root.exists():
        print(f"⚠️  Dataset não encontrado em {dataset_root}")
        print("Crie a estrutura do dataset conforme documentado")
    else:
        # Criar dataset
        dataset = SpectralDataset(dataset_root)

        # Testar uma amostra
        if len(dataset) > 0:
            sample = dataset[0]

            print("\n✅ Sample carregado:")
            print(f"   Video shape: {sample['video'].shape}")
            print(f"   Audio shape: {sample['audio'].shape}")
            print(f"   Sensors shape: {sample['sensors'].shape}")
            print(f"   Label: {sample['label'].item()}")

            # Criar dataloaders
            train_loader, val_loader = create_dataloaders(
                dataset_root,
                batch_size=4
            )

            # Testar batch
            batch = next(iter(train_loader))
            print("\n✅ Batch carregado:")
            print(f"   Video batch: {batch['video'].shape}")
            print(f"   Audio batch: {batch['audio'].shape}")
            print(f"   Sensors batch: {batch['sensors'].shape}")
            print(f"   Labels batch: {batch['label'].shape}")
