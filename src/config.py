# =====================================================================
# 프로젝트 공통 설정 (상수를 한 곳에 모아둠)
# ---------------------------------------------------------------------
# 경로는 이 파일 위치를 기준으로 계산 → 노트북(notebooks/)에서 실행하든
# 루트에서 실행하든(app.py) 항상 맞음.
# =====================================================================
import os

# 프로젝트 루트 (이 파일은 src/config.py → 부모의 부모가 루트)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 멜 스펙트로그램 변환 (MIMII 논문 베이스라인) ---
N_MELS     = 64
N_FFT      = 1024
HOP_LENGTH = 512
SR         = 16000        # MIMII 샘플링 레이트

# --- 학습 하이퍼파라미터 ---
TRAIN_RATIO = 0.8
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 0.001
SEED        = 42

# --- 경로 ---
DATA_ROOT     = os.path.join(ROOT, 'data')
PROCESSED_DIR = os.path.join(DATA_ROOT, 'processed', 'slider')
MODEL_DIR     = os.path.join(ROOT, 'models')

# 소음 레벨별 wav 폴더 (data/slider = -6dB)
NOISE_DIRS = {
    '6dB':  os.path.join(DATA_ROOT, 'slider_6dB'),
    '0dB':  os.path.join(DATA_ROOT, 'slider_0dB'),
    '-6dB': os.path.join(DATA_ROOT, 'slider'),
}
