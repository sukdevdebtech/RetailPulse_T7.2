import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def artifacts(tmp_path_factory):
    """Run the whole pipeline once into a temp dir and share it across tests."""
    from retailpulse import pipeline

    out = tmp_path_factory.mktemp("artifacts")
    pipeline.run(out)
    return out
