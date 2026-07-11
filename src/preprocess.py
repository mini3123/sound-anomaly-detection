# =====================================================================
# 전처리 : wav → 멜 스펙트로그램(dB)
# =====================================================================
import os
import numpy as np
import librosa

from . import config


## -> 함수 기능 : wav 파일 1개를 멜 스펙트로그램(dB) 배열로 변환
## -> 매개 변수 : file_path - wav 경로
## -> 함수 결과 : (N_MELS, 시간프레임) ndarray
def wav_to_mel(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=config.N_MELS,
        n_fft=config.N_FFT, hop_length=config.HOP_LENGTH)
    return librosa.power_to_db(mel)


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
