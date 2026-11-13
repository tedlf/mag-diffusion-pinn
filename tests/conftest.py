import os
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="session")
def results_dir():
    return os.path.join(ROOT, "results")


@pytest.fixture(scope="session")
def data_dir():
    return os.path.join(ROOT, "data")
