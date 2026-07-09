# 소리 기반 설비 이상 감지 프로젝트

프레스 공장 설비의 소리를 듣고 이상(고장 징후)을 감지하는 딥러닝 프로젝트.
자동차 부품사(아진산업)의 프레스 설비 예지보전에 적용하는 것을 목표 시나리오로 한다.

## 프로젝트 개요

- **입력**: 설비 가동 소리 (wav 파일)
- **처리**: 소리 → 스펙트로그램(이미지) 변환 → 딥러닝 모델
- **출력**: 정상 / 이상 판정
- **핵심 아이디어**: 고장 데이터가 거의 없는 실제 공장 상황을 고려해,
  정상 소리만으로 학습하는 이상 탐지(AutoEncoder) 방식을 사용

## 데이터셋: MIMII

히타치가 공개한 산업기계 소리 데이터셋. 4종류 기계(fan, pump, slider, valve)의
정상/이상 소리를 실제 공장 배경소음과 3가지 강도(6dB, 0dB, -6dB SNR)로 섞어 제공.

- 다운로드: https://zenodo.org/records/3384388
- 참고(더 가벼운 버전): DCASE 2020 Task 2 개발용 데이터 https://zenodo.org/records/3678171

**slider(슬라이더)** 사용 (왕복 운동 기계라 프레스와 유사한 스토리).
소음 3레벨(6dB/0dB/-6dB)을 모두 받아 성능 비교까지 완료 (5단계).

> ⚠️ **라벨 정정 (2026-07-08)**: 초기에 "6dB로 시작"이라 적었으나 실제 받아둔 데이터는
> **-6dB(가장 시끄러운 버전)**였음이 파일 해시 대조로 확인됨. 그래서 `data/slider/`(=-6dB) 기준으로
> 1~4단계를 진행했고, 이후 6dB(`data/slider_6dB/`)·0dB(`data/slider_0dB/`)를 추가로 받아 비교했다.

## 데이터 준비 상태 (2026-07-07 압축해제, 2026-07-08 -6dB로 정정)

`data/slider/` 에 **-6dB** 데이터 압축 해제 완료. 기계 4대(id_00, 02, 04, 06), wav 파일 개수:

| 기계 | normal | abnormal |
|------|--------|----------|
| id_00 | 1068 | 356 |
| id_02 | 1068 | 267 |
| id_04 | 534 | 178 |
| id_06 | 534 | 89 |

## 진행 계획

1. [x] 데이터 다운로드 및 소리 데이터 탐색 (파형, 스펙트로그램 그려보기) — 2026-07-07 완료
2. [x] 전처리: 소리 → 멜 스펙트로그램 변환 — 2026-07-07 완료, `data/processed/slider/`에 npy 8개
3. [x] AutoEncoder 모델 학습 (정상 소리만 사용) — 2026-07-08 완료, 이상 복원 오차가 정상의 1.7배로 분리 확인
4. [x] 평가: 이상 소리를 얼마나 잘 잡아내는지 (AUC, 임계값 설정) — 2026-07-08 완료, **-6dB 데이터에서** AUC 0.95 (90퍼센타일 임계값 채택 시 미탐 8/356)
5. [x] 소음 강도별 성능 비교 — 2026-07-09 완료. AUC 6dB 0.998 / 0dB 0.988 / -6dB 0.961 (소음↑ 완만히↓, 최악 조건도 0.96 유지)
6. [x] 정리: 실제 공장 도입 시나리오 + 판정로그 DB(MySQL) + Streamlit 데모 — 2026-07-09 완료

**➡ 진행 계획 1~6단계 전부 완료 (2026-07-09).** 이후 확장 계획은 `docs/확장_로드맵.md` 참고.

## 결과 요약

- **이상 탐지 성능**: AUC **0.95** (-6dB, 가장 시끄러운 조건 기준)
- **소음 강건성**: 6dB **0.998** / 0dB **0.988** / -6dB **0.961** — 소음↑에도 완만한 하락(급락 아님)
- **판정 임계값**: 학습 정상 오차 90퍼센타일 채택 → 이상 356건 중 미탐 8건 (미탐 최소화 우선)
- **검증**: 단순 기준선(스펙트로그램 평균)도 AUC 0.88 → id_00은 비교적 쉬운 케이스이나, AE가 0.95로 개선. 데이터 누수 없음 확인
- **데모**: wav 업로드 → 판정(정상/이상) → 판정 이력 MySQL 기록까지 동작 확인

## 폴더 구조

```
sound_anomaly_project/
├── data/          # MIMII 데이터 (-6dB=slider, slider_6dB, slider_0dB; git 제외)
├── notebooks/     # 실험 노트북 + 단계별 설계 문서(XX_설계.md)
├── models/        # 모델 + 추론용 값 (ae_id00_v1.pth, minmax.npy, threshold.npy)
├── docs/          # 도입 시나리오, 면접 예상질문, 확장 로드맵
├── app.py         # Streamlit 데모 (streamlit run app.py)
├── db_config.py   # MySQL 접속 설정 (비밀번호 포함 → git 제외)
└── README.md
```

## 개발 환경 (2026-07-06 설정 완료)

- conda 환경 **DL_PY311** 사용 (`C:\Users\KDT024\miniconda3\envs\DL_PY311`)
- Python 3.11 / torch 2.11 (CUDA, GPU 사용 가능) / librosa 0.11 설치됨
- 노트북 실행 시 커널을 DL_PY311로 선택할 것

## 노트북 목록

- `00_practice_synthetic.ipynb` : 가짜 기계 소리를 만들어 파형/스펙트로그램 개념 연습 (데이터 불필요)
- `01_data_exploration.ipynb` : MIMII 실제 데이터 탐색 (데이터 다운로드 필요)
- `02_preprocessing.ipynb` : 전체 wav → 멜 스펙트로그램 변환, `data/processed/slider/`에 `기계_라벨.npy` 8개 저장
- `03_autoencoder.ipynb` : AutoEncoder를 정상 소리만으로 학습, `models/ae_id00_v1.pth` 저장 (설계: `03_autoencoder_설계.md`)
- `04_evaluation.ipynb` : 테스트 전체 복원 오차로 AUC(0.95) 계산 + 판정 임계값 설정 (설계: `04_evaluation_설계.md`)
- `05_noise_experiment.ipynb` : 소음 3레벨(6/0/-6dB) AUC 비교 (설계: `05_noise_experiment_설계.md`)
- `06_judgment_log_db.ipynb` : 판정 로그 MySQL DB 저장/조회 (설계: `06_deployment_scenario_설계.md`)

## 문서 (docs/)

- `deployment_scenario.md` : 아진산업 프레스 도입 시나리오 (배경·구성도·운영정책·한계)
- `면접_예상질문.md` : 프로젝트 관련 예상 질문 + 답변 포인트 모음
- `확장_로드맵.md` : 앞으로 할 것 (리팩토링, 진동 데이터 비교 등)

## 데모 실행법

```
conda activate DL_PY311        # 또는 아래처럼 전체 경로로
cd c:\Users\KDT024\Desktop\sound_anomaly_project
streamlit run app.py
```
브라우저에서 10초짜리 wav 업로드 → 정상/이상 판정 + 스펙트로그램 + DB 기록.
(가상환경이 안 잡히면: `C:\Users\KDT024\miniconda3\envs\DL_PY311\python.exe -m streamlit run app.py`)
