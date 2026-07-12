import wave

import numpy as np
import pytest

from src import config
from src.preprocess import sliding_windows, wav_to_mel, wav_to_mel_robust


def write_test_wav(file_path, sr, seconds, silent=False):
    sampleCount = int(sr * seconds)
    if silent:
        yNP = np.zeros(sampleCount, dtype=np.float32)
    else:
        timeNP = np.arange(sampleCount) / sr
        yNP = 0.2 * np.sin(2 * np.pi * 440 * timeNP)

    pcmNP = (yNP * 32767).astype(np.int16)
    with wave.open(str(file_path), 'wb') as wavFile:
        wavFile.setnchannels(1)
        wavFile.setsampwidth(2)
        wavFile.setframerate(sr)
        wavFile.writeframes(pcmNP.tobytes())


@pytest.mark.parametrize(
    'sr,seconds',
    [(8000, 2), (22050, 12)],
)
def test_wav_to_mel_robust_handles_rate_and_length(tmp_path, sr, seconds):
    wavPath = tmp_path / f'test_{sr}_{seconds}.wav'
    write_test_wav(wavPath, sr, seconds)

    mel_dbNP = wav_to_mel_robust(wavPath)

    assert mel_dbNP.shape == (config.N_MELS, 313)
    assert np.isfinite(mel_dbNP).all()


def test_wav_to_mel_robust_keeps_standard_input_result(tmp_path):
    wavPath = tmp_path / 'standard.wav'
    write_test_wav(wavPath, config.SR, 10)

    originalMelNP = wav_to_mel(wavPath)
    robustMelNP = wav_to_mel_robust(wavPath)

    assert originalMelNP.shape == (config.N_MELS, 313)
    assert np.allclose(robustMelNP, originalMelNP)


def test_wav_to_mel_robust_rejects_silence(tmp_path):
    wavPath = tmp_path / 'silence.wav'
    write_test_wav(wavPath, config.SR, 10, silent=True)

    with pytest.raises(ValueError, match='무음'):
        wav_to_mel_robust(wavPath)


@pytest.mark.parametrize('bad_value', [np.nan, np.inf])
def test_wav_to_mel_robust_rejects_nonfinite_value(monkeypatch, bad_value):
    # 실제 WAV로 만들기 어려운 NaN/Inf 입력을 librosa 반환값 단계에서 재현한다.
    def fake_load(file_path, sr):
        return np.array([bad_value], dtype=np.float32), sr

    monkeypatch.setattr('src.preprocess.librosa.load', fake_load)

    with pytest.raises(ValueError, match='NaN 또는 무한대'):
        wav_to_mel_robust('fake.wav')


def test_wav_to_mel_robust_rejects_empty_wav(tmp_path):
    wavPath = tmp_path / 'empty.wav'
    write_test_wav(wavPath, config.SR, 0)

    with pytest.raises(ValueError, match='빈 WAV'):
        wav_to_mel_robust(wavPath)


def test_wav_to_mel_robust_rejects_broken_file(tmp_path):
    brokenPath = tmp_path / 'broken.wav'
    brokenPath.write_text('wav 파일이 아님', encoding='utf-8')

    with pytest.raises(ValueError, match='불러올 수 없습니다'):
        wav_to_mel_robust(brokenPath)


def test_sliding_windows_40_seconds_count():
    yNP = np.arange(40)

    windowLST = sliding_windows(yNP, win_len=10, hop_len=1)

    assert len(windowLST) == 31
    assert np.array_equal(windowLST[0], np.arange(0, 10))
    assert np.array_equal(windowLST[-1], np.arange(30, 40))


def test_sliding_windows_drops_incomplete_last_window():
    yNP = np.arange(25)

    windowLST = sliding_windows(yNP, win_len=10, hop_len=10)

    assert len(windowLST) == 2
    assert np.array_equal(windowLST[-1], np.arange(10, 20))


def test_sliding_windows_short_signal_returns_empty_list():
    yNP = np.arange(5)

    windowLST = sliding_windows(yNP, win_len=10, hop_len=1)

    assert windowLST == []


@pytest.mark.parametrize(
    'win_len,hop_len',
    [(0, 1), (-1, 1), (10, 0), (10, -1)],
)
def test_sliding_windows_rejects_invalid_length(win_len, hop_len):
    yNP = np.arange(40)

    with pytest.raises(ValueError):
        sliding_windows(yNP, win_len=win_len, hop_len=hop_len)
