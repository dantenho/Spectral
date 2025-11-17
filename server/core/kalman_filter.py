"""
Filtro de Kalman para suavização e predição de dados de sensores
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class KalmanState:
    """Estado do filtro de Kalman"""
    x: np.ndarray  # Estado estimado
    P: np.ndarray  # Matriz de covariância do erro
    timestamp: float


class KalmanFilter1D:
    """
    Filtro de Kalman 1D otimizado para sensores

    Uso: Suavizar magnitude do magnetômetro, remover ruído, predizer valores
    """

    def __init__(
        self,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-2,
        initial_value: float = 0.0
    ):
        """
        Args:
            process_variance: Variância do processo (Q) - quanto o sistema varia
            measurement_variance: Variância da medição (R) - ruído do sensor
            initial_value: Valor inicial estimado
        """
        self.Q = process_variance  # Process noise
        self.R = measurement_variance  # Measurement noise

        # Estado inicial
        self.x = initial_value  # Estimativa
        self.P = 1.0  # Incerteza

        self.initialized = False

    def predict(self, dt: float = 1.0) -> float:
        """
        Predição do próximo estado

        Args:
            dt: Delta time (não usado em modelo estacionário)

        Returns:
            Valor predito
        """
        # Modelo de predição: x(k) = x(k-1) (modelo estacionário)
        # A covariância aumenta devido ao ruído do processo
        self.P = self.P + self.Q

        return self.x

    def update(self, measurement: float) -> float:
        """
        Atualização com nova medição

        Args:
            measurement: Nova medição do sensor

        Returns:
            Valor filtrado
        """
        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x

        # Ganho de Kalman
        K = self.P / (self.P + self.R)

        # Atualizar estimativa
        self.x = self.x + K * (measurement - self.x)

        # Atualizar covariância
        self.P = (1 - K) * self.P

        return self.x

    def process(self, measurement: float, dt: float = 1.0) -> float:
        """
        Predição + Atualização em um passo

        Args:
            measurement: Nova medição
            dt: Delta time

        Returns:
            Valor filtrado
        """
        self.predict(dt)
        return self.update(measurement)


class KalmanFilter3D:
    """
    Filtro de Kalman 3D para vetores (ex: campo magnético xyz)

    Suaviza cada componente independentemente
    """

    def __init__(
        self,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-2,
        initial_value: Optional[np.ndarray] = None
    ):
        """
        Args:
            process_variance: Variância do processo
            measurement_variance: Variância da medição
            initial_value: Valor inicial [x, y, z]
        """
        if initial_value is None:
            initial_value = np.zeros(3)

        # Criar 3 filtros independentes
        self.filters = [
            KalmanFilter1D(process_variance, measurement_variance, initial_value[i])
            for i in range(3)
        ]

    def process(self, measurement: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """
        Processa vetor 3D

        Args:
            measurement: Vetor [x, y, z]
            dt: Delta time

        Returns:
            Vetor filtrado [x, y, z]
        """
        return np.array([
            self.filters[i].process(measurement[i], dt)
            for i in range(3)
        ])


class AdaptiveKalmanFilter:
    """
    Filtro de Kalman adaptativo que ajusta automaticamente
    a covariância do ruído baseado na variância das medições

    Útil quando o ruído do sensor varia com o tempo
    """

    def __init__(
        self,
        process_variance: float = 1e-5,
        initial_measurement_variance: float = 1e-2,
        adaptation_rate: float = 0.1,
        window_size: int = 10
    ):
        """
        Args:
            process_variance: Variância do processo (Q)
            initial_measurement_variance: Variância inicial (R)
            adaptation_rate: Taxa de adaptação (0-1)
            window_size: Janela para calcular variância
        """
        self.Q = process_variance
        self.R = initial_measurement_variance
        self.adaptation_rate = adaptation_rate
        self.window_size = window_size

        self.x = 0.0
        self.P = 1.0

        # Buffer para calcular variância adaptativa
        from collections import deque
        self.measurement_buffer = deque(maxlen=window_size)

        self.initialized = False

    def _update_measurement_variance(self):
        """Atualiza R baseado na variância das medições recentes"""
        if len(self.measurement_buffer) < 3:
            return

        variance = np.var(self.measurement_buffer)

        # Adaptação suave
        self.R = (1 - self.adaptation_rate) * self.R + self.adaptation_rate * variance

    def process(self, measurement: float, dt: float = 1.0) -> float:
        """Processa medição com adaptação"""
        self.measurement_buffer.append(measurement)

        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x

        # Atualizar R adaptativamente
        self._update_measurement_variance()

        # Predict
        self.P = self.P + self.Q

        # Update
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P

        return self.x


class ExtendedKalmanFilter:
    """
    Filtro de Kalman Estendido (EKF) para sistemas não-lineares

    Útil para fusão de sensores com modelos complexos
    """

    def __init__(
        self,
        state_dim: int = 3,
        measurement_dim: int = 3,
        process_variance: float = 1e-5,
        measurement_variance: float = 1e-2
    ):
        """
        Args:
            state_dim: Dimensão do estado
            measurement_dim: Dimensão da medição
            process_variance: Variância do processo
            measurement_variance: Variância da medição
        """
        self.n = state_dim
        self.m = measurement_dim

        # Estado
        self.x = np.zeros(state_dim)
        self.P = np.eye(state_dim)

        # Ruído
        self.Q = np.eye(state_dim) * process_variance
        self.R = np.eye(measurement_dim) * measurement_variance

        self.initialized = False

    def predict(self, F: np.ndarray, dt: float = 1.0):
        """
        Predição com matriz de transição F

        Args:
            F: Matriz de transição de estado (n x n)
            dt: Delta time
        """
        # x(k) = F * x(k-1)
        self.x = F @ self.x

        # P(k) = F * P(k-1) * F^T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: np.ndarray, H: np.ndarray):
        """
        Atualização com medição z e matriz de observação H

        Args:
            z: Vetor de medição (m x 1)
            H: Matriz de observação (m x n)
        """
        if not self.initialized:
            # Inicializar com pseudo-inversa
            self.x = np.linalg.pinv(H) @ z
            self.initialized = True
            return

        # Inovação: y = z - H * x
        y = z - H @ self.x

        # Covariância da inovação: S = H * P * H^T + R
        S = H @ self.P @ H.T + self.R

        # Ganho de Kalman: K = P * H^T * S^-1
        K = self.P @ H.T @ np.linalg.inv(S)

        # Atualizar estado: x = x + K * y
        self.x = self.x + K @ y

        # Atualizar covariância: P = (I - K * H) * P
        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P


# ============================================================================
# EXEMPLO DE USO
# ============================================================================

def example_usage():
    """Exemplo de uso dos filtros de Kalman"""

    # 1. Filtro 1D para magnitude magnética
    kalman_mag = KalmanFilter1D(
        process_variance=1e-5,
        measurement_variance=5e-2
    )

    # Simular medições com ruído
    true_value = 50.0
    measurements = true_value + np.random.normal(0, 5, 100)

    filtered_values = []
    for measurement in measurements:
        filtered = kalman_mag.process(measurement)
        filtered_values.append(filtered)

    print(f"Medição média: {np.mean(measurements):.2f}")
    print(f"Filtrado médio: {np.mean(filtered_values):.2f}")
    print(f"Valor real: {true_value:.2f}")

    # 2. Filtro 3D para vetor magnético
    kalman_3d = KalmanFilter3D(
        process_variance=1e-5,
        measurement_variance=1e-2
    )

    # Simular vetor com ruído
    true_vector = np.array([20.0, -15.0, 45.0])
    for _ in range(50):
        noisy_measurement = true_vector + np.random.normal(0, 2, 3)
        filtered_vector = kalman_3d.process(noisy_measurement)

    print(f"\nVetor filtrado: {filtered_vector}")
    print(f"Vetor real: {true_vector}")

    # 3. Filtro adaptativo
    adaptive_kalman = AdaptiveKalmanFilter(
        process_variance=1e-5,
        initial_measurement_variance=1e-2,
        adaptation_rate=0.1
    )

    # Simular mudança no ruído ao longo do tempo
    for i in range(100):
        noise_level = 2.0 if i < 50 else 5.0  # Ruído aumenta
        measurement = true_value + np.random.normal(0, noise_level)
        filtered = adaptive_kalman.process(measurement)

    print(f"\nR adaptado: {adaptive_kalman.R:.4f}")


if __name__ == "__main__":
    example_usage()
