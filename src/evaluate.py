# =====================================================================
# 평가 : 복원 오차 계산, AUC, 임계값
# =====================================================================
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


## -> 함수 기능 : 배열 전체의 샘플별 복원 오차
## -> 매개 변수 : model, dataNP - (개수,64,313), device
## -> 함수 결과 : (개수,) 오차 배열
def recon_errors(model, dataNP, device):
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(dataNP.reshape(len(dataNP), -1)).to(device)
        out = model(x)
        err = ((x - out) ** 2).mean(dim=1)   # 샘플별 평균
    return err.cpu().numpy()


## -> 함수 기능 : 정상/이상 오차로 AUC 계산
def compute_auc(err_normal, err_abnormal):
    y_true = np.array([0] * len(err_normal) + [1] * len(err_abnormal))
    y_score = np.concatenate([err_normal, err_abnormal])
    return roc_auc_score(y_true, y_score)


## -> 함수 기능 : 학습 정상 오차의 percentile로 판정 임계값
def make_threshold(err_train, percentile=90):
    return np.percentile(err_train, percentile)


## -> 함수 기능 : 해당 기계일 확률을 이용해 분류 기반 이상탐지 AUC 계산
## -> 매개 변수 : model, machine_idx, test_normalNP, test_abnormalNP, device
## -> 함수 결과 : 이상점수(1 - 해당 기계 확률)로 계산한 ROC AUC
def classifier_auc(model, machine_idx, test_normalNP, test_abnormalNP, device):
    """이상점수 = 1 - softmax(logits)[machine_idx]. 기계별 AUC 반환."""
    model.eval()

    # 입력 배열을 모델 확률로 바꾼 뒤, 해당 기계가 아닐 확률을 이상점수로 사용한다.
    def score(a):
        with torch.no_grad():
            x = torch.FloatTensor(a.reshape(len(a), -1)).to(device)
            p = torch.softmax(model(x), dim=1)[:, machine_idx]
        return (1 - p).cpu().numpy()

    s_normal = score(test_normalNP)
    s_abnormal = score(test_abnormalNP)
    y_true = np.array([0]*len(s_normal) + [1]*len(s_abnormal))
    y_score = np.concatenate([s_normal, s_abnormal])
    return roc_auc_score(y_true, y_score)


## -> 함수 기능 : 멜 스펙트로그램마다 4개 기계 ID의 softmax 확률 계산
## -> 매개 변수 : model, dataNP - (개수,64,313), device
## -> 함수 결과 : (개수,4) softmax 확률 ndarray
def predict_proba(model, dataNP, device):
    """분류 모델의 logits를 기계별 softmax 확률로 변환해 반환."""
    model.eval()
    with torch.no_grad():
        # 기존 classifier_auc와 같은 방식으로 입력을 한 줄 벡터로 편다.
        x = torch.FloatTensor(dataNP.reshape(len(dataNP), -1)).to(device)
        proba = torch.softmax(model(x), dim=1)
    return proba.cpu().numpy()


## -> 함수 기능 : 멜 스펙트로그램마다 분류 직전 128차원 특징 추출
## -> 매개 변수 : model, dataNP - (개수,64,313), device, batch_size
## -> 함수 결과 : (개수,128) 임베딩 ndarray
def extract_embeddings(model, dataNP, device, batch_size=128):
    """IDClassifier의 마지막 Linear 직전 임베딩을 배치 단위로 반환."""
    model.eval()
    embedding_list = []

    # 전체 데이터를 한꺼번에 GPU에 올리지 않도록 작은 배치로 나눈다.
    with torch.no_grad():
        for start in range(0, len(dataNP), batch_size):
            end = start + batch_size
            batchNP = dataNP[start:end]
            x = torch.FloatTensor(batchNP.reshape(len(batchNP), -1)).to(device)
            embedding = model.extract_embedding(x)
            embedding_list.append(embedding.cpu().numpy())

    if len(embedding_list) == 0:
        return np.empty((0, 128), dtype=np.float32)
    return np.concatenate(embedding_list, axis=0)
