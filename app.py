# =====================================================================
# 프레스 설비 소리 이상 감지 - 분류 앙상블 Streamlit 데모
# 실행: 프로젝트 루트에서 streamlit run app.py
# =====================================================================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from datetime import datetime

import librosa
import librosa.display
import matplotlib.pyplot as plt
import koreanize_matplotlib
import numpy as np
import pymysql
import streamlit as st
import torch

from db_config import DB_CONFIG
from src.config import HOP_LENGTH, MODEL_DIR, N_FFT, N_MELS
from src.evaluate import predict_proba
from src.model import IDClassifier


MACHINES = ('id_00', 'id_02', 'id_04', 'id_06')
MODEL_SEEDS = [0, 1, 2, 3, 4]
DEVICE = 'cpu'


# 모델과 전처리 기준은 앱 실행 중 한 번만 불러온다.
@st.cache_resource
def load_all():
    modelLST = []
    for seed in MODEL_SEEDS:
        model = IDClassifier()
        model_path = os.path.join(MODEL_DIR, f'idclf_{seed}.pth')
        model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        model.eval()
        modelLST.append(model)

    MIN, MAX = np.load(os.path.join(MODEL_DIR, 'clf_minmax.npy'))
    thrNP = np.load(os.path.join(MODEL_DIR, 'clf_thresholds.npy'))
    return modelLST, MIN, MAX, thrNP


modelLST, MIN, MAX, thrNP = load_all()

st.title('프레스 설비 소리 이상 감지')
machine = st.selectbox('설비 ID 선택', MACHINES)
machine_idx = MACHINES.index(machine)
uploaded = st.file_uploader('설비 소리 wav 파일 업로드', type='wav')

if uploaded is not None:
    y, sr = librosa.load(uploaded, sr=None)
    st.audio(uploaded)
    mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )
    )

    # 학습 데이터와 같은 10초·16 kHz 입력만 판정한다.
    if mel_db.shape[1] != 313:
        st.warning(f'이 데모는 10초(16 kHz) wav 전용입니다. (현재 프레임 {mel_db.shape[1]}개)')
        st.stop()

    xNP = ((mel_db - MIN) / (MAX - MIN)).reshape(1, -1)

    # 다섯 초기값 모델의 샘플별 이상점수를 평균한다.
    scoreSUM = 0.0
    for model in modelLST:
        probaNP = predict_proba(model, xNP, DEVICE)
        scoreSUM = scoreSUM + (1 - probaNP[0, machine_idx])
    score = scoreSUM / len(modelLST)

    THRESHOLD = thrNP[machine_idx]
    judgment = 'abnormal' if score > THRESHOLD else 'normal'

    st.write(f'이상점수 : {score:.4f} / 임계값 : {THRESHOLD:.4f} (설비 {machine})')
    if judgment == 'abnormal':
        st.error('이상! 점검이 필요합니다.')
    else:
        st.success('정상입니다.')

    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(
        mel_db,
        sr=sr,
        hop_length=HOP_LENGTH,
        x_axis='time',
        y_axis='mel',
        ax=ax,
    )
    ax.set_title(f'{machine} 멜 스펙트로그램')
    ax.set_xlabel('시간')
    ax.set_ylabel('멜 주파수')
    st.pyplot(fig)

    # 기존 DB 스키마를 유지하고 error 열에는 앙상블 이상점수를 기록한다.
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute(
        'INSERT INTO judgment_log(created_at, machine_id, error, judgment) '
        'VALUES(%s,%s,%s,%s)',
        (datetime.now(), machine, score, judgment),
    )
    conn.commit()
    conn.close()
    st.caption('판정 결과를 DB에 기록했습니다.')
