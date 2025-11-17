import pytest
import numpy as np
import sys
import os

# Add the server directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processing.audio_variants import (
    FFTVariant,
    STFTVariant,
    WaveletVariant,
    FormantVariant,
    FilterbankVariant,
    ZeroCrossingVariant,
    AudioEnsemble
)

# Test data generation
def generate_test_audio(
    duration=1.0,
    sample_rate=44100,
    freq=440.0,
    amplitude=0.1,
    base_signal='sine',
    add_infrasound=False,
    add_ultrasound=False,
    add_transient=False,
    add_noise_component=False
):
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    if base_signal == 'sine':
        audio = amplitude * np.sin(2 * np.pi * freq * t)
    elif base_signal == 'noise':
        audio = amplitude * np.random.randn(num_samples)
    else:  # 'zeros'
        audio = np.zeros(num_samples)

    if add_infrasound:
        audio += 0.5 * np.sin(2 * np.pi * 10 * t)

    if add_ultrasound:
        audio += 0.5 * np.sin(2 * np.pi * 20000 * t)

    if add_transient:
        start = num_samples // 2
        end = start + 100
        audio[start:end] += 0.8 * np.random.randn(100)

    if add_noise_component:
        audio += 0.05 * np.random.randn(num_samples)

    return audio

# Tests for FFTVariant
def test_fft_variant_no_anomaly():
    audio = generate_test_audio(amplitude=0.001)
    variant = FFTVariant(anomaly_threshold_db=30)
    result = variant.analyze(audio)
    assert not result.anomaly_detected

def test_fft_variant_infrasound_anomaly():
    audio = generate_test_audio(base_signal='zeros', add_infrasound=True)
    variant = FFTVariant()
    result = variant.analyze(audio)
    assert result.anomaly_detected

def test_fft_variant_ultrasound_anomaly():
    audio = generate_test_audio(base_signal='zeros', add_ultrasound=True)
    variant = FFTVariant()
    result = variant.analyze(audio)
    assert result.anomaly_detected

def test_fft_variant_peak_power_anomaly():
    audio = generate_test_audio(freq=1000, amplitude=1.0)
    variant = FFTVariant(anomaly_threshold_db=-5)
    result = variant.analyze(audio)
    assert result.anomaly_detected

# Tests for STFTVariant
def test_stft_variant_no_anomaly():
    audio = generate_test_audio()
    variant = STFTVariant()
    result = variant.analyze(audio)
    assert not result.anomaly_detected

def test_stft_variant_transient_anomaly():
    audio = generate_test_audio(add_transient=True)
    variant = STFTVariant(transient_threshold=0.1)
    result = variant.analyze(audio)
    assert result.anomaly_detected

# Tests for WaveletVariant
def test_wavelet_variant_no_anomaly():
    audio = generate_test_audio()
    variant = WaveletVariant()
    result = variant.analyze(audio)
    assert not result.anomaly_detected

def test_wavelet_variant_low_freq_anomaly():
    audio = generate_test_audio(freq=50)
    variant = WaveletVariant(energy_threshold=0.1)
    result = variant.analyze(audio)
    assert result.anomaly_detected

def test_wavelet_variant_high_freq_anomaly():
    audio = generate_test_audio(freq=18000)
    variant = WaveletVariant()
    result = variant.analyze(audio)
    assert result.anomaly_detected

# Tests for FormantVariant
def test_formant_variant_no_anomaly():
    audio = generate_test_audio(freq=1000)
    variant = FormantVariant()
    result = variant.analyze(audio)
    assert not result.anomaly_detected

# Tests for FilterbankVariant
def test_filterbank_variant_no_anomaly():
    audio = generate_test_audio(base_signal='noise')
    variant = FilterbankVariant(anomaly_threshold_db=0)
    result = variant.analyze(audio)
    assert not result.anomaly_detected

def test_filterbank_variant_high_energy_anomaly():
    audio = generate_test_audio(freq=2000)
    variant = FilterbankVariant(anomaly_threshold_db=-5)
    result = variant.analyze(audio)
    assert result.anomaly_detected

# Tests for ZeroCrossingVariant
def test_zero_crossing_variant_no_anomaly():
    audio = generate_test_audio(freq=1000)
    variant = ZeroCrossingVariant()
    result = variant.analyze(audio)
    assert not result.anomaly_detected

def test_zero_crossing_variant_high_zcr_anomaly():
    audio = generate_test_audio(base_signal='noise')
    variant = ZeroCrossingVariant()
    result = variant.analyze(audio)
    assert result.anomaly_detected

# Tests for AudioEnsemble
def test_audio_ensemble():
    audio = generate_test_audio(add_infrasound=True, add_transient=True)
    ensemble = AudioEnsemble()
    result = ensemble.analyze(audio)
    assert result['anomaly_detected']
    assert result['num_anomalies_detected'] > 0

def test_audio_ensemble_no_anomaly():
    audio = generate_test_audio()
    ensemble = AudioEnsemble()
    result = ensemble.analyze(audio)
    assert not result['anomaly_detected']
