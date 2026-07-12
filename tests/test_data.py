import numpy as np
import pytest

from src.data import load_for_classification, load_split_normalize


MACHINES = ('id_00', 'id_02', 'id_04', 'id_06')


@pytest.fixture
def fake_processed_dir(tmp_path):
    # seed 42에서 테스트로 빠지는 index 3/6에 학습 범위 밖 값을 둔다.
    for machine_idx in range(len(MACHINES)):
        machine = MACHINES[machine_idx]
        normalNP = np.zeros((10, 64, 313), dtype=np.float32)
        for sample_idx in range(10):
            normalNP[sample_idx] = machine_idx * 10 + sample_idx
        normalNP[3] = -100
        normalNP[6] = 1000

        abnormalNP = np.full(
            (5, 64, 313), machine_idx * 10 + 20, dtype=np.float32)
        np.save(tmp_path / f'{machine}_normal.npy', normalNP)
        np.save(tmp_path / f'{machine}_abnormal.npy', abnormalNP)
    return tmp_path


def test_load_split_normalize_shape_and_train_range(fake_processed_dir):
    trainNP, testNormalNP, testAbnormalNP = load_split_normalize(
        seed=42, machine='id_00', processed_dir=fake_processed_dir)

    assert trainNP.shape == (8, 64, 313)
    assert testNormalNP.shape == (2, 64, 313)
    assert testAbnormalNP.shape == (5, 64, 313)
    assert np.isclose(trainNP.min(), 0.0)
    assert np.isclose(trainNP.max(), 1.0)
    # 테스트 값이 0~1 밖에 남아 있어 train 기준으로만 정규화했음을 확인한다.
    assert testNormalNP.min() < 0
    assert testNormalNP.max() > 1


def test_load_for_classification_return_values(fake_processed_dir):
    result = load_for_classification(
        seed=42, processed_dir=fake_processed_dir, return_minmax=True)
    X_trainNP, y_trainNP, tests, MIN, MAX = result

    assert X_trainNP.shape == (32, 64, 313)
    assert y_trainNP.shape == (32,)
    assert set(np.unique(y_trainNP)) == {0, 1, 2, 3}
    assert len(tests) == 4
    assert MIN == 0
    assert MAX == 39
    assert X_trainNP.min() == 0
    assert X_trainNP.max() == 1

    compatible_result = load_for_classification(
        seed=42, processed_dir=fake_processed_dir)
    assert len(compatible_result) == 3
