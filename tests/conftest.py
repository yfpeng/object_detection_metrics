from pathlib import Path
from typing import Dict

import pytest
import numpy as np
from podm.podm import MetricPerClass


@pytest.fixture(scope="module")
def example_dir():
    return Path(__file__).parent / '../examples/'


@pytest.fixture(scope="module")
def tests_dir():
    return Path(__file__).parent

