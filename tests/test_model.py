import torch

from src.model import AutoEncoder, IDClassifier


def test_autoencoder_output_shape_and_range():
    model = AutoEncoder()
    inputTensor = torch.rand(2, 20032)

    outputTensor = model(inputTensor)

    assert outputTensor.shape == (2, 20032)
    assert torch.all(outputTensor >= 0)
    assert torch.all(outputTensor <= 1)


def test_id_classifier_output_shape():
    model = IDClassifier()
    inputTensor = torch.rand(2, 20032)

    outputTensor = model(inputTensor)

    assert outputTensor.shape == (2, 4)


def test_id_classifier_embedding_shape_and_forward_compatibility():
    model = IDClassifier()
    inputTensor = torch.rand(2, 20032)

    embeddingTensor = model.extract_embedding(inputTensor)
    outputTensor = model(inputTensor)

    assert embeddingTensor.shape == (2, 128)
    assert outputTensor.shape == (2, 4)
