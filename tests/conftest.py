import os, sys, tempfile
os.environ["FACE_ENGINE"] = "mock"
os.environ.setdefault("DATA_DIR", tempfile.mkdtemp())
os.environ["DEV"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pytest


@pytest.fixture(autouse=True)
def _reset_ip_limits():
    import app as A
    A._ip_hits.clear()
    saved = dict(A.IP_LIMITS)
    yield
    A.IP_LIMITS.update(saved)
    A._ip_hits.clear()


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    """Every test gets its own empty database and uploads folder (no cross-test or order dependence)."""
    import app as A
    (tmp_path / "uploads").mkdir()
    monkeypatch.setattr(A, "DATA", str(tmp_path))
    monkeypatch.setattr(A, "UPLOADS", str(tmp_path / "uploads"))
    monkeypatch.setattr(A, "DB_PATH", str(tmp_path / "app.db"))
    A.init_db()
    yield
