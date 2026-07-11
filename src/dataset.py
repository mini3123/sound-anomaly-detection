# =====================================================================
# Dataset : 멜 스펙트로그램 배열을 AutoEncoder 학습용으로 제공
# ---------------------------------------------------------------------
# AutoEncoder는 정답 = 입력 자신이라 피쳐만 반환한다.
# =====================================================================
import torch
from torch.utils.data import Dataset


class MelDataset(Dataset):
    def __init__(self, dataNP):
        super().__init__()
        self.data = dataNP.reshape(len(dataNP), -1)   # (개수, 64, 313) → (개수, 20032)
        self.rows = len(dataNP)

    def __len__(self):
        return self.rows

    def __getitem__(self, index):
        return torch.FloatTensor(self.data[index])
