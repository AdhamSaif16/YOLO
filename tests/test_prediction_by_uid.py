import os
import sqlite3
from fastapi.testclient import TestClient
from app import app, DB_PATH, save_prediction_session

client = TestClient(app)

def setup_module(module):
    # Insert a test prediction directly into the DB
    module.uid = "test-uid-123"
    module.original = "uploads/original/test.jpg"
    module.predicted = "uploads/predicted/test.jpg"
    os.makedirs("uploads/original", exist_ok=True)
    os.makedirs("uploads/predicted", exist_ok=True)
    open(module.original, 'a').close()
    open(module.predicted, 'a').close()

    save_prediction_session(module.uid, module.original, module.predicted)

def teardown_module(module):
    # Clean up DB and files
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (module.uid,))
    os.remove(module.original)
    os.remove(module.predicted)

def test_get_prediction_by_uid_success():
    response = client.get("/prediction/test-uid-123")
    assert response.status_code == 200
    data = response.json()
    assert data["uid"] == "test-uid-123"
    assert "original_image" in data
    assert "predicted_image" in data

def test_get_prediction_by_uid_not_found():
    response = client.get("/prediction/nonexistent-uid")
    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found"
