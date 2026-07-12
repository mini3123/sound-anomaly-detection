# =====================================================================
# 모델 : 완전연결 AutoEncoder
# ---------------------------------------------------------------------
# 구조 20032(=64x313) → 512 → 64 → 512 → 20032
# 입력을 압축(encoder)했다가 복원(decoder). 정상만 학습 → 이상은 복원 오차 큼.
# =====================================================================
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20032, 512), nn.ReLU(),
            nn.Linear(512, 64),    nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 512),    nn.ReLU(),
            nn.Linear(512, 20032), nn.Sigmoid(),   # 입력이 0~1이라 출력도 0~1
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class CNNAutoEncoder(nn.Module):
    """스펙트로그램 2D 구조를 살리는 Conv AutoEncoder.
    입출력은 Linear AE와 동일하게 펴진 벡터(b,20032) → 기존 파이프라인 드롭인 교체."""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  nn.ReLU(),   # 64x313 → 32x157
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # → 16x79
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # → 8x40
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=(1, 0)), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=(1, 0)), nn.ReLU(),
            nn.ConvTranspose2d(16, 1,  3, stride=2, padding=1, output_padding=(1, 0)), nn.Sigmoid(),
        )

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, 1, 64, 313)              # 펴진 벡터 → 2D 이미지
        out = self.decoder(self.encoder(x))
        return out.view(b, -1)                 # 다시 펴서 반환


class CNNBottleneckAE(nn.Module):
    """conv로 특징 뽑고 Linear로 latent까지 강하게 압축한 CNN AutoEncoder.
    병목 없는 CNNAutoEncoder가 이상까지 잘 복원해 탐지력이 낮았던 문제를 해결."""
    def __init__(self, latent=64):
        super().__init__()
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  nn.ReLU(),   # 64x313 → 32x157
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),   # → 16x79
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # → 8x40
        )
        self.flat_dim = 64 * 8 * 40            # 20480
        self.enc_fc = nn.Linear(self.flat_dim, latent)    # 진짜 병목: 20480 → latent
        self.dec_fc = nn.Linear(latent, self.flat_dim)
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=(1, 0)), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=(1, 0)), nn.ReLU(),
            nn.ConvTranspose2d(16, 1,  3, stride=2, padding=1, output_padding=(1, 0)), nn.Sigmoid(),
        )

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, 1, 64, 313)
        z = self.enc_fc(self.enc_conv(x).view(b, -1))     # (b, latent) 병목
        h = self.dec_fc(z).view(b, 64, 8, 40)
        out = self.dec_conv(h)
        return out.view(b, -1)


class IDClassifier(nn.Module):
    """멜스펙 → 기계 ID(4-class) 분류. 인코더는 CNNBottleneckAE와 동일 conv."""
    def __init__(self, num_classes=4):
        super().__init__()
        # 멜 스펙트로그램의 시간-주파수 패턴을 3단계 CNN으로 추출한다.
        self.enc_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
        )
        # 추출한 특징을 128차원으로 줄인 뒤 4개 기계의 점수(logits)를 출력한다.
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 40, 128), nn.ReLU(),
            nn.Linear(128, num_classes),          # logits (softmax는 손실/점수에서)
        )

    def extract_embedding(self, x):
        """마지막 분류층 직전의 128차원 특징을 반환한다."""
        b = x.shape[0]
        x = x.view(b, 1, 64, 313)
        z = self.enc_conv(x).view(b, -1)
        return self.fc[1](self.fc[0](z))           # (b, 128)

    def forward(self, x):
        embedding = self.extract_embedding(x)
        return self.fc[2](embedding)                # (b, num_classes)
