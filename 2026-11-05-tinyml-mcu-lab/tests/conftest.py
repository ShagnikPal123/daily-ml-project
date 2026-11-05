import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinyml_lab.experiments import Context  # noqa: E402


@pytest.fixture(scope="session")
def ctx():
    return Context(seeds=(0,), epochs=40)
