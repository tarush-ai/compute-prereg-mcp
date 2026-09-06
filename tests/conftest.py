import warnings
from pathlib import Path

import pytest

from compute_prereg.broker.adapters.mock import MockAdapter
from compute_prereg.engine import Engine, EngineConfig

warnings.filterwarnings("ignore", category=UserWarning)

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "tiny_transformer.py"


@pytest.fixture
def example() -> Path:
    return EXAMPLE


@pytest.fixture
def mock_adapter() -> MockAdapter:
    return MockAdapter()


@pytest.fixture
def engine(tmp_path, mock_adapter):
    cfg = EngineConfig(data_dir=tmp_path / "data", poll_interval=0.2, confirmation_ttl=60)
    eng = Engine(cfg, adapters=[mock_adapter], in_memory=True)
    eng.executor.checkpoint_grace = 2.0
    eng.executor.profile_every, eng.executor.profile_steps = 10, 3
    yield eng
    eng.close()


def make_prereg(engine: Engine, example: Path, *, total_steps=40, mode="ask", **env_overrides):
    est = engine.estimate_job(entrypoint=str(example), gpu_candidates=["H100", "A100-80GB", "RTX 4090"],
                              total_steps=total_steps)
    env = Engine.suggest_envelope(est, "H100-SXM-80GB")
    env = env.model_copy(update={"spot_ok": True, **env_overrides})
    return est, engine.preregister(est.estimate_id, env.model_dump(), mode)
