from fastapi.testclient import TestClient

from pv_triage.api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_triage_endpoint_returns_redacted_report(sample):
    resp = client.post("/triage", json={"text": sample("02_liver_failure_unexpected.txt")})
    assert resp.status_code == 200
    body = resp.json()
    assert body["triage"]["priority"] == "expedited"
    assert body["phi_redacted"]["EMAIL"] == 1
    assert "p.natarajan@example-pharmacy.com" not in resp.text
    assert "Priya Natarajan" not in resp.text


def test_rejects_short_input():
    assert client.post("/triage", json={"text": "short"}).status_code == 422
