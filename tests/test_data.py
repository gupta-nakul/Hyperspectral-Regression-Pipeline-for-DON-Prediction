import pytest
from src.data.dataset_loader import DatasetLoader

def test_dataset_loader():
    loader = DatasetLoader("tests/test_data.csv")
    X, y = loader.get_data()
    assert len(X) == len(y), "Number of features and targets should match"
    # Add more checks...
