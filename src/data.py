# =====================================================================
# 데이터 준비 : 전처리된 npy 로드 → 학습/테스트 분리 → 정규화
# ---------------------------------------------------------------------
# 03·04·05에 반복되던 "로드→분리→정규화"를 함수 하나로.
# =====================================================================
import os
import numpy as np

from . import config


## -> 함수 기능 : id_XX_normal/abnormal.npy 로드 → seed로 8:2 분리 → 학습 기준 0~1 정규화
## -> 매개 변수 : seed(분리 재현), machine, processed_dir
## -> 함수 결과 : trainNP, test_normalNP, test_abnormalNP (모두 0~1 정규화됨)
def load_split_normalize(seed=config.SEED, machine='id_00', processed_dir=None):
    pdir = processed_dir if processed_dir else config.PROCESSED_DIR
    normalNP   = np.load(os.path.join(pdir, f'{machine}_normal.npy'))
    abnormalNP = np.load(os.path.join(pdir, f'{machine}_abnormal.npy'))

    np.random.seed(seed)
    idx = np.random.permutation(len(normalNP))
    train_num = int(len(normalNP) * config.TRAIN_RATIO)
    trainNP       = normalNP[idx[:train_num]]
    test_normalNP = normalNP[idx[train_num:]]

    MIN, MAX = trainNP.min(), trainNP.max()       # 기준은 학습 데이터에서만
    trainNP         = (trainNP - MIN) / (MAX - MIN)
    test_normalNP   = (test_normalNP - MIN) / (MAX - MIN)
    test_abnormalNP = (abnormalNP - MIN) / (MAX - MIN)
    return trainNP, test_normalNP, test_abnormalNP


## -> 함수 기능 : 4대 정상 데이터를 합쳐 기계 ID 분류용 학습/테스트 데이터 준비
## -> 매개 변수 : seed, machines - 기계 ID 목록, processed_dir
## -> 함수 결과 : X_train, y_train, tests - 전역 MIN/MAX로 정규화된 데이터
def load_for_classification(seed=config.SEED, machines=('id_00','id_02','id_04','id_06'),
                            processed_dir=None, return_minmax=False):
    """4대 정상을 합쳐 학습셋(X,y) + 기계별 테스트셋. 전역 MIN/MAX로 정규화."""
    pdir = processed_dir if processed_dir else config.PROCESSED_DIR
    train_list, y_list, tests = [], [], {}
    raw_train = []   # 정규화 전 train_normal 모으기(전역 MIN/MAX용)
    per_machine = []

    # 각 기계의 정상 데이터를 같은 seed로 80% 학습, 20% 테스트로 나눈다.
    for i in range(len(machines)):
        m = machines[i]
        normalNP = np.load(os.path.join(pdir, f'{m}_normal.npy'))
        abnormalNP = np.load(os.path.join(pdir, f'{m}_abnormal.npy'))
        np.random.seed(seed)
        idx = np.random.permutation(len(normalNP))
        tn = int(len(normalNP) * config.TRAIN_RATIO)
        train_normal = normalNP[idx[:tn]]
        test_normal = normalNP[idx[tn:]]
        raw_train.append(train_normal)
        per_machine.append((i, train_normal, test_normal, abnormalNP))

    # 한 분류 모델이 4대를 함께 보므로 학습 정상 전체의 공통 범위를 사용한다.
    MIN = np.concatenate(raw_train).min()
    MAX = np.concatenate(raw_train).max()

    def norm(a): return (a - MIN) / (MAX - MIN)

    # 기계 index를 정답 라벨로 붙이고, 기계별 테스트 데이터는 따로 보관한다.
    for i, train_normal, test_normal, abnormalNP in per_machine:
        train_list.append(norm(train_normal))
        y_list.append(np.full(len(train_normal), i))
        tests[machines[i]] = (i, norm(test_normal), norm(abnormalNP))

    X_train = np.concatenate(train_list)
    y_train = np.concatenate(y_list)
    if return_minmax:
        return X_train, y_train, tests, MIN, MAX
    return X_train, y_train, tests
