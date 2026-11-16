import pytest

from desk.config import Settings, load_settings
from desk.store import Store


@pytest.fixture
def settings(tmp_path) -> Settings:
    # Anthropic provider so the Claude-caller tests keep their model defaults; demo agents stay scripted.
    return load_settings({"DESK_MODE": "demo", "DESK_DB": str(tmp_path / "t.db"), "DESK_AUTOSTART": "0",
                          "LLM_PROVIDER": "anthropic"}, use_dotenv=False)


@pytest.fixture
def store(tmp_path) -> Store:
    s = Store(str(tmp_path / "s.db"))
    yield s
    s.close()
