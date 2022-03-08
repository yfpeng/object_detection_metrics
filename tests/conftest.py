from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def example_dir():
    return Path(__file__).parent / '../examples/'


@pytest.fixture(scope="module")
def tests_dir():
    return Path(__file__).parent

