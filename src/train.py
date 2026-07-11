# =====================================================================
# 학습 : 정상 데이터로 AutoEncoder 학습
# =====================================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import config
from .dataset import MelDataset
from .model import AutoEncoder


## -> 함수 기능 : 넘겨받은 모델을 정상 데이터로 학습 (Linear/CNN 등 아무 모델)
## -> 매개 변수 : model - 학습할 모델(이미 .to(device)됨), trainNP - 0~1 정규화 정상, device
## -> 함수 결과 : 학습된 model, loss_history
def train_model(model, trainNP, device):
    loader = DataLoader(MelDataset(trainNP), batch_size=config.BATCH_SIZE, shuffle=True)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    loss_history = []
    for epoch in range(config.EPOCHS):
        model.train()
        losses = []
        for feature in loader:
            feature = feature.to(device)
            out = model(feature)
            optimizer.zero_grad()
            loss = loss_fn(out, feature)   # 정답 = 입력 자신
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_history.append(sum(losses) / len(losses))
    return model, loss_history


## -> 편의 함수 : 기본 Linear AutoEncoder를 만들어 학습 (기존 노트북 호환)
def train_autoencoder(trainNP, device):
    return train_model(AutoEncoder().to(device), trainNP, device)


## -> 함수 기능 : 정상 소리와 기계 ID 라벨로 분류 모델 학습
## -> 매개 변수 : model, X_NP - 정규화 정상 멜스펙, y_NP - 기계 index, device
## -> 함수 결과 : 학습된 기계 ID 분류 모델
def train_classifier(model, X_NP, y_NP, device):
    """X_NP:(n,64,313) 정규화 정상, y_NP:(n,) 기계 index. CrossEntropy 학습."""
    from torch.utils.data import TensorDataset, DataLoader

    # CNN 입력 규격에 맞게 멜 스펙트로그램을 한 줄 벡터로 펴고 Tensor로 바꾼다.
    X = torch.FloatTensor(X_NP.reshape(len(X_NP), -1))
    y = torch.LongTensor(y_NP)
    loader = DataLoader(TensorDataset(X, y), batch_size=config.BATCH_SIZE, shuffle=True)

    # 다중 클래스 분류이므로 CrossEntropyLoss를 사용한다.
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    for epoch in range(config.EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            optimizer.zero_grad()
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
    return model
