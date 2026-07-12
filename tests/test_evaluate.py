import numpy as np

from src.evaluate import (
    compute_auc,
    extract_embeddings,
    make_threshold,
    predict_proba,
    recon_errors,
)
from src.model import AutoEncoder, IDClassifier


def test_compute_auc_perfect_separation():
    normalScoreNP = np.array([0.1, 0.2, 0.3])
    abnormalScoreNP = np.array([0.7, 0.8, 0.9])

    auc = compute_auc(normalScoreNP, abnormalScoreNP)

    assert auc == 1.0


def test_compute_auc_same_distribution():
    normalScoreNP = np.array([0.1, 0.2, 0.3])
    abnormalScoreNP = np.array([0.1, 0.2, 0.3])

    auc = compute_auc(normalScoreNP, abnormalScoreNP)

    assert np.isclose(auc, 0.5)


def test_make_threshold_90_percentile():
    trainErrorNP = np.arange(0, 101)

    threshold = make_threshold(trainErrorNP, percentile=90)

    assert threshold == 90


def test_predict_proba_shape_and_row_sum():
    model = IDClassifier()
    dataNP = np.random.default_rng(42).random((3, 20032)).astype(np.float32)

    probaNP = predict_proba(model, dataNP, 'cpu')

    assert probaNP.shape == (3, 4)
    assert np.allclose(probaNP.sum(axis=1), 1.0)


def test_recon_errors_shape():
    model = AutoEncoder()
    dataNP = np.random.default_rng(42).random((3, 20032)).astype(np.float32)

    errorNP = recon_errors(model, dataNP, 'cpu')

    assert errorNP.shape == (3,)


def test_extract_embeddings_shape_and_batching():
    model = IDClassifier()
    dataNP = np.random.default_rng(42).random((3, 20032)).astype(np.float32)

    embeddingNP = extract_embeddings(model, dataNP, 'cpu', batch_size=2)

    assert embeddingNP.shape == (3, 128)
    assert np.isfinite(embeddingNP).all()


def test_extract_embeddings_empty_input():
    model = IDClassifier()
    dataNP = np.empty((0, 20032), dtype=np.float32)

    embeddingNP = extract_embeddings(model, dataNP, 'cpu')

    assert embeddingNP.shape == (0, 128)
