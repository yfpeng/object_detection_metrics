from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def sample_dir():
    return Path(__file__).parent / 'sample'


@pytest.fixture(scope="module")
def tests_dir():
    return Path(__file__).parent

