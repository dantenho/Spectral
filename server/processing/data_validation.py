"""
Sistema de Validação de Dados

Valida integridade, qualidade e consistência de dados sensoriais
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import re
from datetime import datetime


# ============================================================================
# VALIDATION RESULTS
# ============================================================================

class ValidationSeverity(Enum):
    """Severidade da validação"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Issue de validação"""
    field: str
    message: str
    severity: ValidationSeverity
    value: Any = None
    expected: Any = None


@dataclass
class ValidationResult:
    """Resultado de validação"""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    warnings: int = 0
    errors: int = 0
    critical: int = 0

    def add_issue(self, issue: ValidationIssue):
        """Adiciona issue"""
        self.issues.append(issue)

        if issue.severity == ValidationSeverity.WARNING:
            self.warnings += 1
        elif issue.severity == ValidationSeverity.ERROR:
            self.errors += 1
            self.is_valid = False
        elif issue.severity == ValidationSeverity.CRITICAL:
            self.critical += 1
            self.is_valid = False

    def to_dict(self) -> Dict:
        """Converte para dicionário"""
        return {
            'is_valid': self.is_valid,
            'warnings': self.warnings,
            'errors': self.errors,
            'critical': self.critical,
            'issues': [
                {
                    'field': issue.field,
                    'message': issue.message,
                    'severity': issue.severity.value,
                    'value': str(issue.value) if issue.value is not None else None,
                    'expected': str(issue.expected) if issue.expected is not None else None
                }
                for issue in self.issues
            ]
        }


# ============================================================================
# SENSOR DATA VALIDATOR
# ============================================================================

class SensorDataValidator:
    """
    Valida dados de sensores

    Verifica:
    - Ranges físicos válidos
    - Tipos de dados
    - Valores ausentes
    - Outliers extremos
    - Taxas de mudança
    """

    # Ranges físicos para cada tipo de sensor
    SENSOR_RANGES = {
        'accelerometer': {
            'min': -20.0,  # m/s²
            'max': 20.0,
            'typical_range': 9.81,  # Gravidade terrestre
            'max_rate_of_change': 100.0  # m/s² por segundo
        },
        'gyroscope': {
            'min': -35.0,  # rad/s (≈2000 deg/s)
            'max': 35.0,
            'typical_range': 8.7,  # 500 deg/s
            'max_rate_of_change': 50.0
        },
        'magnetometer': {
            'min': -100.0,  # µT
            'max': 100.0,
            'typical_range': 50.0,  # Campo magnético terrestre
            'max_rate_of_change': 10.0
        },
        'pressure': {
            'min': 30000.0,  # Pa (≈300 mbar, topo Mt. Everest)
            'max': 110000.0,  # Pa (≈1100 mbar, ciclone)
            'typical_range': 101325.0,  # Nível do mar
            'max_rate_of_change': 1000.0
        },
        'temperature': {
            'min': -40.0,  # °C
            'max': 85.0,   # °C (smartphone operating range)
            'typical_range': 25.0,
            'max_rate_of_change': 10.0  # Por minuto
        },
        'light': {
            'min': 0.0,  # lux
            'max': 100000.0,  # Luz solar direta
            'typical_range': 400.0,  # Ambiente interno
            'max_rate_of_change': 50000.0
        },
        'proximity': {
            'min': 0.0,  # cm
            'max': 10.0,
            'typical_range': 5.0,
            'max_rate_of_change': 100.0
        }
    }

    def __init__(self):
        self.previous_values = {}

    def validate_sensor_packet(self, packet: Dict) -> ValidationResult:
        """
        Valida pacote completo de sensores

        Args:
            packet: Dict com dados dos sensores
                {
                    'timestamp': float,
                    'sensors': {
                        'accelerometer': {'x': float, 'y': float, 'z': float},
                        'gyroscope': {'x': float, 'y': float, 'z': float},
                        ...
                    }
                }

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # 1. Validar timestamp
        if 'timestamp' not in packet:
            result.add_issue(ValidationIssue(
                field='timestamp',
                message='Timestamp ausente',
                severity=ValidationSeverity.CRITICAL
            ))
        else:
            self._validate_timestamp(packet['timestamp'], result)

        # 2. Validar estrutura
        if 'sensors' not in packet:
            result.add_issue(ValidationIssue(
                field='sensors',
                message='Campo sensors ausente',
                severity=ValidationSeverity.CRITICAL
            ))
            return result

        sensors = packet['sensors']

        # 3. Validar cada sensor
        for sensor_name, sensor_data in sensors.items():
            self._validate_sensor(sensor_name, sensor_data, result)

        return result

    def _validate_timestamp(self, timestamp: float, result: ValidationResult):
        """Valida timestamp"""

        # Verificar tipo
        if not isinstance(timestamp, (int, float)):
            result.add_issue(ValidationIssue(
                field='timestamp',
                message='Timestamp deve ser numérico',
                severity=ValidationSeverity.ERROR,
                value=type(timestamp).__name__
            ))
            return

        # Verificar range razoável (últimos 10 anos)
        min_timestamp = datetime(2015, 1, 1).timestamp()
        max_timestamp = datetime(2030, 1, 1).timestamp()

        if timestamp < min_timestamp or timestamp > max_timestamp:
            result.add_issue(ValidationIssue(
                field='timestamp',
                message='Timestamp fora do range válido',
                severity=ValidationSeverity.ERROR,
                value=timestamp,
                expected=f'[{min_timestamp}, {max_timestamp}]'
            ))

        # Verificar monotonia (se houver timestamp anterior)
        if hasattr(self, 'last_timestamp'):
            if timestamp <= self.last_timestamp:
                result.add_issue(ValidationIssue(
                    field='timestamp',
                    message='Timestamp não é monotonicamente crescente',
                    severity=ValidationSeverity.WARNING,
                    value=timestamp,
                    expected=f'> {self.last_timestamp}'
                ))

        self.last_timestamp = timestamp

    def _validate_sensor(
        self,
        sensor_name: str,
        sensor_data: Dict,
        result: ValidationResult
    ):
        """Valida dados de um sensor específico"""

        # Obter configuração do sensor
        sensor_config = self.SENSOR_RANGES.get(sensor_name)

        if sensor_config is None:
            # Sensor desconhecido - apenas warning
            result.add_issue(ValidationIssue(
                field=sensor_name,
                message=f'Sensor desconhecido: {sensor_name}',
                severity=ValidationSeverity.INFO
            ))
            return

        # Validar estrutura (espera-se x, y, z ou value)
        if not isinstance(sensor_data, dict):
            result.add_issue(ValidationIssue(
                field=sensor_name,
                message='Dados do sensor devem ser um dicionário',
                severity=ValidationSeverity.ERROR,
                value=type(sensor_data).__name__
            ))
            return

        # Validar componentes
        if 'x' in sensor_data and 'y' in sensor_data and 'z' in sensor_data:
            # Sensor 3D
            for axis in ['x', 'y', 'z']:
                value = sensor_data[axis]
                field_name = f'{sensor_name}.{axis}'
                self._validate_value(
                    field_name, value, sensor_config, result, sensor_name
                )

        elif 'value' in sensor_data:
            # Sensor escalar
            value = sensor_data['value']
            self._validate_value(
                sensor_name, value, sensor_config, result, sensor_name
            )

        else:
            result.add_issue(ValidationIssue(
                field=sensor_name,
                message='Sensor deve ter componentes x,y,z ou value',
                severity=ValidationSeverity.ERROR,
                value=list(sensor_data.keys())
            ))

    def _validate_value(
        self,
        field_name: str,
        value: Any,
        config: Dict,
        result: ValidationResult,
        sensor_name: str
    ):
        """Valida um valor individual"""

        # 1. Tipo
        if not isinstance(value, (int, float)):
            result.add_issue(ValidationIssue(
                field=field_name,
                message='Valor deve ser numérico',
                severity=ValidationSeverity.ERROR,
                value=type(value).__name__
            ))
            return

        # 2. NaN/Inf
        if np.isnan(value) or np.isinf(value):
            result.add_issue(ValidationIssue(
                field=field_name,
                message='Valor inválido (NaN ou Inf)',
                severity=ValidationSeverity.ERROR,
                value=value
            ))
            return

        # 3. Range físico
        min_val = config['min']
        max_val = config['max']

        if value < min_val or value > max_val:
            result.add_issue(ValidationIssue(
                field=field_name,
                message='Valor fora do range físico',
                severity=ValidationSeverity.ERROR,
                value=value,
                expected=f'[{min_val}, {max_val}]'
            ))

        # 4. Outlier extremo (>5x typical range)
        typical = config['typical_range']
        if abs(value) > typical * 5:
            result.add_issue(ValidationIssue(
                field=field_name,
                message='Outlier extremo detectado',
                severity=ValidationSeverity.WARNING,
                value=value,
                expected=f'|value| < {typical * 5}'
            ))

        # 5. Taxa de mudança
        prev_key = f'{field_name}_prev'
        if prev_key in self.previous_values:
            prev_value = self.previous_values[prev_key]
            rate_of_change = abs(value - prev_value)
            max_rate = config['max_rate_of_change']

            if rate_of_change > max_rate:
                result.add_issue(ValidationIssue(
                    field=field_name,
                    message='Taxa de mudança muito alta',
                    severity=ValidationSeverity.WARNING,
                    value=rate_of_change,
                    expected=f'< {max_rate}'
                ))

        self.previous_values[prev_key] = value


# ============================================================================
# SCHEMA VALIDATOR
# ============================================================================

class SchemaValidator:
    """
    Valida estrutura de dados contra schema

    Schema format:
    {
        'field_name': {
            'type': str | int | float | bool | list | dict,
            'required': bool,
            'min': Optional[float],
            'max': Optional[float],
            'pattern': Optional[str],  # regex
            'enum': Optional[List],
            'nested_schema': Optional[Dict]
        }
    }
    """

    @staticmethod
    def validate(data: Dict, schema: Dict) -> ValidationResult:
        """Valida dados contra schema"""

        result = ValidationResult(is_valid=True)

        # 1. Validar campos obrigatórios
        for field_name, field_spec in schema.items():
            required = field_spec.get('required', False)

            if required and field_name not in data:
                result.add_issue(ValidationIssue(
                    field=field_name,
                    message='Campo obrigatório ausente',
                    severity=ValidationSeverity.ERROR
                ))

        # 2. Validar campos presentes
        for field_name, value in data.items():
            if field_name not in schema:
                result.add_issue(ValidationIssue(
                    field=field_name,
                    message='Campo não esperado',
                    severity=ValidationSeverity.INFO,
                    value=value
                ))
                continue

            field_spec = schema[field_name]
            SchemaValidator._validate_field(
                field_name, value, field_spec, result
            )

        return result

    @staticmethod
    def _validate_field(
        field_name: str,
        value: Any,
        spec: Dict,
        result: ValidationResult
    ):
        """Valida um campo individual"""

        # 1. Tipo
        expected_type = spec.get('type')
        if expected_type is not None:
            if not isinstance(value, expected_type):
                result.add_issue(ValidationIssue(
                    field=field_name,
                    message='Tipo incorreto',
                    severity=ValidationSeverity.ERROR,
                    value=type(value).__name__,
                    expected=expected_type.__name__
                ))
                return

        # 2. Enum
        enum_values = spec.get('enum')
        if enum_values is not None:
            if value not in enum_values:
                result.add_issue(ValidationIssue(
                    field=field_name,
                    message='Valor não está no enum',
                    severity=ValidationSeverity.ERROR,
                    value=value,
                    expected=enum_values
                ))

        # 3. Min/Max (numérico)
        if isinstance(value, (int, float)):
            min_val = spec.get('min')
            max_val = spec.get('max')

            if min_val is not None and value < min_val:
                result.add_issue(ValidationIssue(
                    field=field_name,
                    message='Valor abaixo do mínimo',
                    severity=ValidationSeverity.ERROR,
                    value=value,
                    expected=f'>= {min_val}'
                ))

            if max_val is not None and value > max_val:
                result.add_issue(ValidationIssue(
                    field=field_name,
                    message='Valor acima do máximo',
                    severity=ValidationSeverity.ERROR,
                    value=value,
                    expected=f'<= {max_val}'
                ))

        # 4. Pattern (string)
        if isinstance(value, str):
            pattern = spec.get('pattern')
            if pattern is not None:
                if not re.match(pattern, value):
                    result.add_issue(ValidationIssue(
                        field=field_name,
                        message='Valor não corresponde ao pattern',
                        severity=ValidationSeverity.ERROR,
                        value=value,
                        expected=pattern
                    ))

        # 5. Nested schema (dict)
        if isinstance(value, dict):
            nested_schema = spec.get('nested_schema')
            if nested_schema is not None:
                nested_result = SchemaValidator.validate(value, nested_schema)

                for issue in nested_result.issues:
                    # Prefixar com field name
                    issue.field = f'{field_name}.{issue.field}'
                    result.add_issue(issue)


# ============================================================================
# CONSISTENCY VALIDATOR
# ============================================================================

class ConsistencyValidator:
    """
    Valida consistência entre múltiplos campos e sensores
    """

    @staticmethod
    def validate_cross_field_consistency(data: Dict) -> ValidationResult:
        """
        Valida consistência entre campos relacionados

        Exemplos:
        - Magnitude calculada vs componentes
        - Relações físicas entre sensores
        """
        result = ValidationResult(is_valid=True)

        # 1. Verificar magnitude vs componentes
        if 'sensors' in data:
            sensors = data['sensors']

            for sensor_name in ['accelerometer', 'gyroscope', 'magnetometer']:
                if sensor_name in sensors:
                    sensor_data = sensors[sensor_name]

                    if all(k in sensor_data for k in ['x', 'y', 'z', 'magnitude']):
                        x, y, z = sensor_data['x'], sensor_data['y'], sensor_data['z']
                        mag_reported = sensor_data['magnitude']

                        # Calcular magnitude esperada
                        mag_calculated = np.sqrt(x**2 + y**2 + z**2)

                        # Tolerância de 1%
                        if abs(mag_calculated - mag_reported) > 0.01 * mag_calculated:
                            result.add_issue(ValidationIssue(
                                field=f'{sensor_name}.magnitude',
                                message='Magnitude inconsistente com componentes',
                                severity=ValidationSeverity.WARNING,
                                value=mag_reported,
                                expected=mag_calculated
                            ))

        # 2. Verificar consistência temporal
        if 'timestamp' in data and 'frame_number' in data:
            # Frame rate esperado (~30 fps)
            # Timestamp deve aumentar consistentemente

            pass  # Requer histórico, implementar em classe com estado

        return result


# ============================================================================
# BATCH VALIDATOR
# ============================================================================

class BatchValidator:
    """
    Valida lotes de dados (múltiplos pacotes)
    """

    @staticmethod
    def validate_batch(packets: List[Dict]) -> ValidationResult:
        """Valida lote de pacotes"""

        result = ValidationResult(is_valid=True)

        if not packets:
            result.add_issue(ValidationIssue(
                field='batch',
                message='Lote vazio',
                severity=ValidationSeverity.WARNING
            ))
            return result

        # 1. Verificar ordem temporal
        timestamps = [p.get('timestamp', 0) for p in packets]

        if not all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)):
            result.add_issue(ValidationIssue(
                field='batch.timestamps',
                message='Pacotes fora de ordem temporal',
                severity=ValidationSeverity.WARNING
            ))

        # 2. Verificar gaps temporais grandes
        if len(timestamps) > 1:
            diffs = np.diff(timestamps)
            median_diff = np.median(diffs)

            # Gaps > 10x mediana
            large_gaps = np.where(diffs > median_diff * 10)[0]

            if len(large_gaps) > 0:
                result.add_issue(ValidationIssue(
                    field='batch.timestamps',
                    message=f'Encontrados {len(large_gaps)} gaps temporais grandes',
                    severity=ValidationSeverity.WARNING,
                    value=len(large_gaps)
                ))

        # 3. Verificar duplicatas
        unique_timestamps = len(set(timestamps))
        if unique_timestamps < len(timestamps):
            result.add_issue(ValidationIssue(
                field='batch.timestamps',
                message='Timestamps duplicados encontrados',
                severity=ValidationSeverity.WARNING,
                value=f'{len(timestamps) - unique_timestamps} duplicados'
            ))

        return result


# ============================================================================
# TESTE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TESTE DE VALIDAÇÃO DE DADOS")
    print("=" * 70)

    # 1. Sensor Data Validator
    print("\n1. Sensor Data Validator:")

    validator = SensorDataValidator()

    # Pacote válido
    valid_packet = {
        'timestamp': datetime.now().timestamp(),
        'sensors': {
            'accelerometer': {'x': 0.5, 'y': -0.3, 'z': 9.8},
            'gyroscope': {'x': 0.1, 'y': -0.05, 'z': 0.02},
            'magnetometer': {'x': 25.0, 'y': -30.0, 'z': 45.0}
        }
    }

    result = validator.validate_sensor_packet(valid_packet)
    print(f"  Valid packet: {result.is_valid}")
    print(f"  Issues: {len(result.issues)}")

    # Pacote inválido
    invalid_packet = {
        'timestamp': datetime.now().timestamp(),
        'sensors': {
            'accelerometer': {'x': 100.0, 'y': -0.3, 'z': 9.8},  # x fora do range
            'gyroscope': {'x': 'invalid', 'y': -0.05, 'z': 0.02}  # tipo errado
        }
    }

    result = validator.validate_sensor_packet(invalid_packet)
    print(f"\n  Invalid packet: {result.is_valid}")
    print(f"  Errors: {result.errors}")
    print(f"  Warnings: {result.warnings}")

    for issue in result.issues:
        print(f"    - {issue.field}: {issue.message} (severity: {issue.severity.value})")

    # 2. Schema Validator
    print("\n2. Schema Validator:")

    schema = {
        'device_id': {
            'type': str,
            'required': True,
            'pattern': r'^[A-Za-z0-9\-]+$'
        },
        'version': {
            'type': str,
            'required': True
        },
        'battery_level': {
            'type': float,
            'required': False,
            'min': 0.0,
            'max': 100.0
        }
    }

    data = {
        'device_id': 'device-123',
        'version': '1.0.0',
        'battery_level': 85.5
    }

    schema_validator = SchemaValidator()
    result = schema_validator.validate(data, schema)

    print(f"  Valid: {result.is_valid}")
    print(f"  Issues: {len(result.issues)}")

    # 3. Batch Validator
    print("\n3. Batch Validator:")

    batch = [
        {'timestamp': 1000.0},
        {'timestamp': 1001.0},
        {'timestamp': 1002.0},
        {'timestamp': 1020.0},  # Gap grande
        {'timestamp': 1021.0}
    ]

    batch_validator = BatchValidator()
    result = batch_validator.validate_batch(batch)

    print(f"  Valid: {result.is_valid}")
    print(f"  Warnings: {result.warnings}")

    for issue in result.issues:
        print(f"    - {issue.field}: {issue.message}")
