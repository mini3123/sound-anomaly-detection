# =====================================================================
# 전처리 : wav → 멜 스펙트로그램(dB)
# =====================================================================
import os
import numpy as np
import librosa

from . import config


TARGET_SECONDS = 10
TARGET_SAMPLES = config.SR * TARGET_SECONDS
EXPECTED_FRAMES = 1 + TARGET_SAMPLES // config.HOP_LENGTH


## -> 함수 기능 : wav 파일 1개를 멜 스펙트로그램(dB) 배열로 변환
## -> 매개 변수 : file_path - wav 경로
## -> 함수 결과 : (N_MELS, 시간프레임) ndarray
def wav_to_mel(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=config.N_MELS,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    return librosa.power_to_db(mel)


## -> 함수 기능 : 길이·샘플레이트가 다른 wav를 10초·16kHz 고정 mel로 변환
## -> 매개 변수 : file_path - wav 경로 또는 파일 객체
## -> 함수 결과 : 항상 (N_MELS, EXPECTED_FRAMES)인 mel dB ndarray
def wav_to_mel_robust(file_path):
    # 휴대폰 녹음처럼 샘플레이트가 달라도 학습 조건과 같은 16kHz로 맞춘다.
    try:
        yNP, sr = librosa.load(file_path, sr=config.SR)
    except Exception as error:
        raise ValueError(f'WAV 파일을 불러올 수 없습니다: {error}') from error

    # 빈 파일·무음·비정상 숫자는 모델에 넣어도 의미 있는 판정을 할 수 없다.
    if len(yNP) == 0:
        raise ValueError('빈 WAV 파일은 판정할 수 없습니다.')
    if not np.isfinite(yNP).all():
        raise ValueError('WAV 파일에 NaN 또는 무한대 값이 포함되어 있습니다.')
    if np.max(np.abs(yNP)) == 0:
        raise ValueError('무음 WAV 파일은 판정할 수 없습니다.')

    # 짧으면 뒤를 0으로 채우고, 길면 앞의 10초만 사용한다.
    if len(yNP) < TARGET_SAMPLES:
        padding = TARGET_SAMPLES - len(yNP)
        yNP = np.pad(yNP, (0, padding), mode='constant')
    elif len(yNP) > TARGET_SAMPLES:
        yNP = yNP[:TARGET_SAMPLES]

    melNP = librosa.feature.melspectrogram(
        y=yNP, sr=sr, n_mels=config.N_MELS,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    mel_dbNP = librosa.power_to_db(melNP)

    # 고정 길이 처리 뒤에도 예상 shape이 아니면 명확히 중단한다.
    expected_shape = (config.N_MELS, EXPECTED_FRAMES)
    if mel_dbNP.shape != expected_shape:
        raise ValueError(
            f'멜 스펙트로그램 shape이 예상과 다릅니다: '
            f'{mel_dbNP.shape} != {expected_shape}')
    return mel_dbNP


## -> 함수 기능 : 연속 신호를 고정 길이의 겹치는 창으로 분할
## -> 매개 변수 : yNP - 1차원 소리 배열, win_len - 창 길이, hop_len - 이동 길이
## -> 함수 결과 : 완전한 길이를 가진 창 ndarray의 리스트(마지막 불완전 창 제외)
def sliding_windows(yNP, win_len, hop_len):
    if win_len <= 0:
        raise ValueError('win_len은 0보다 커야 합니다.')
    if hop_len <= 0:
        raise ValueError('hop_len은 0보다 커야 합니다.')

    windowLST = []
    last_start = len(yNP) - win_len
    if last_start < 0:
        return windowLST

    for start in range(0, last_start + 1, hop_len):
        windowNP = yNP[start:start + win_len]
        windowLST.append(windowNP)
    return windowLST


## -> 함수 기능 : 한 폴더(기계/라벨)의 wav 전부를 mel 배열로
## -> 매개 변수 : data_dir - 소음레벨 폴더, label - normal/abnormal, machine_id
## -> 함수 결과 : (파일수, N_MELS, 시간프레임) ndarray
def load_mel(data_dir, label, machine_id='id_00'):
    folder = os.path.join(data_dir, machine_id, label)
    filelist = os.listdir(folder)
    filelist.sort()
    mel_list = []
    for i in range(len(filelist)):
        mel_list.append(wav_to_mel(os.path.join(folder, filelist[i])))
    return np.array(mel_list)
