# =====================================================================
# 프레스 설비 소리 이상 감지 - Streamlit 데모
# 실행: (프로젝트 루트에서) streamlit run app.py
# =====================================================================
# ✍️ 타이핑 연습용 뼈대. 막히면 notebooks/06_deployment_scenario_설계.md [3] 참고.
# 흐름: wav 업로드 → 멜 스펙트로그램 → 정규화 → 복원 오차 → 판정 → 표시 + DB 기록
# 준비물(이미 있음): models/ae_id00_v1.pth, models/minmax.npy, models/threshold.npy

## [0] 모듈
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from datetime import datetime
import pymysql
import streamlit as st
from db_config import DB_CONFIG

N_MELS, N_FFT, HOP_LENGTH = 64, 1024, 512

## [1] 모델 클래스 (03과 동일하게 복사)
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(20032,512), nn.ReLU(), nn.Linear(512,64), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(64,512), nn.ReLU(), nn.Linear(512,20032), nn.Sigmoid())
    def forward(self, x): return self.decoder(self.encoder(x))

## [2] 추론 3종 세트 불러오기 (한 번만)
@st.cache_resource
def load_all():
    model = AutoEncoder()
    model.load_state_dict(torch.load('models/ae_id00_v1.pth', weights_only=True))
    model.eval()
    MIN,MAX = np.load('models/minmax.npy')
    THRESHOLD = np.load('models/threshold.npy')[0]
    return model, MIN, MAX, THRESHOLD
model, MIN, MAX, THRESHOLD = load_all()
## [3] 화면 - 제목 + 파일 업로드
st.title('프레스 설비 소리 이상 감지')
uploaded = st.file_uploader('설비 소리 wav 파일 업로드',type='wav')

## [4] 업로드되면 판정
if uploaded is not None:
    y, sr = librosa.load(uploaded, sr=None)
    st.audio(uploaded)
    mel_db = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y,sr=sr,n_mels=N_MELS, n_fft=N_FFT
                                       ,hop_length=HOP_LENGTH
                                       ))
    if mel_db.shape[1] != 313:
        st.warning(f'이 데모는 10초(16kHz) wav 전용입니다. (현재 프레임 {mel_db.shape[1]}개)')
        st.stop()
        
    x = torch.FloatTensor(((mel_db - MIN) / (MAX -MIN)).reshape(1, -1))
    
    with torch.no_grad():
        err = ((x-model(x))**2).mean().item()
        
    judgment = 'abnormal' if err > THRESHOLD else 'normal'
        
    st.write(f'복원 오차 : {err:.6f} / 임계값 : {THRESHOLD:.6f}')
    if judgment == 'abnormal':
        st.error('⚠️ 이상! 점검이 필요합니다')
    else:
        st.success('✅ 정상입니다.')
        
    fig, ax =plt.subplots(figsize=(10,4))
    librosa.display.specshow(mel_db,sr=sr,hop_length=HOP_LENGTH,x_axis='time',
                             y_axis='mel', ax=ax)
    st.pyplot(fig)
    
    conn = pymysql.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute('INSERT INTO judgment_log(created_at, machine_id, error, judgment)VALUES(%s,%s,%s,%s)',
                (datetime.now(), 'demo',err,judgment))
    conn.commit()
    conn.close()
    st.caption('판정 결과를 DB에 기록했습니다.')