import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def forbid_http_requests(monkeypatch):
    """Collectors in the offline suite must receive an explicit fake session."""
    def reject_request(*args, **kwargs):
        raise AssertionError("HTTP requests are forbidden in the test suite")

    monkeypatch.setattr(requests.sessions.Session, "request", reject_request)
