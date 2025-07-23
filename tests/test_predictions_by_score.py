import os
import sqlite3
import pytest
from fastapi.testclient import TestClient
from app import app, DB_PATH, init_db, add_test_user

client = TestClient(app)

UID_HIGH = "score-high"
UID_LOW = "score-low"
UID_BORDERLINE = "score-border"
SCORE_HIGH = 0.95
SCORE_LOW = 0.4
SCORE_BORDER = 0.7


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    init_db()
    add_test_user()

    with sqlite3.connect(DB_PATH) as conn:
        # Insert prediction sessions
        conn.execute("INSERT OR IGNORE INTO prediction_sessions (uid, original_image, predicted_image, user_id) VALUES (?, ?, ?, ?)",
                     (UID_HIGH, "o1.jpg", "p1.jpg", 1))
        conn.execute("INSERT OR IGNORE INTO prediction_sessions (uid, original_image, predicted_image, user_id) VALUES (?, ?, ?, ?)",
                     (UID_LOW, "o2.jpg", "p2.jpg", 1))
        conn.execute("INSERT OR IGNORE INTO prediction_sessions (uid, original_image, predicted_image, user_id) VALUES (?, ?, ?, ?)",
                     (UID_BORDERLINE, "o3.jpg", "p3.jpg", 1))

        # Insert detection objects
        conn.execute("INSERT INTO detection_objects (prediction_uid, label, score, box) VALUES (?, ?, ?, ?)",
                     (UID_HIGH, "dog", SCORE_HIGH, "[1,1,2,2]"))
        conn.execute("INSERT INTO detection_objects (prediction_uid, label, score, box) VALUES (?, ?, ?, ?)",
                     (UID_LOW, "cat", SCORE_LOW, "[3,3,4,4]"))
        conn.execute("INSERT INTO detection_objects (prediction_uid, label, score, box) VALUES (?, ?, ?, ?)",
                     (UID_BORDERLINE, "horse", SCORE_BORDER, "[5,5,6,6]"))

    yield

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM detection_objects WHERE prediction_uid IN (?, ?, ?)", (UID_HIGH, UID_LOW, UID_BORDERLINE))
        conn.execute("DELETE FROM prediction_sessions WHERE uid IN (?, ?, ?)", (UID_HIGH, UID_LOW, UID_BORDERLINE))


def test_get_predictions_above_0_5():
    response = client.get("/predictions/score/0.5", auth=("user", "pass"))
    assert response.status_code == 200
    uids = [item["uid"] for item in response.json()]
    assert UID_HIGH in uids
    assert UID_BORDERLINE in uids
    assert UID_LOW not in uids


def test_get_predictions_above_0_9():
    response = client.get("/predictions/score/0.9", auth=("user", "pass"))
    assert response.status_code == 200
    uids = [item["uid"] for item in response.json()]
    assert UID_HIGH in uids
    assert UID_LOW not in uids
    assert UID_BORDERLINE not in uids


def test_get_predictions_invalid_score_negative():
    response = client.get("/predictions/score/-0.1", auth=("user", "pass"))
    assert response.status_code == 400
    assert "Invalid score" in response.text


def test_get_predictions_invalid_score_over_1():
    response = client.get("/predictions/score/1.1", auth=("user", "pass"))
    assert response.status_code == 400
    assert "Invalid score" in response.text
