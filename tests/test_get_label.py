import os
import sqlite3
import pytest
from fastapi.testclient import TestClient
from app import app, DB_PATH, init_db, add_test_user

client = TestClient(app)

UID_DOG = "label-dog-1"
UID_CAT = "label-cat-1"


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    init_db()
    add_test_user()

    with sqlite3.connect(DB_PATH) as conn:
        # Insert prediction sessions
        conn.execute("INSERT INTO prediction_sessions (uid, original_image, predicted_image, user_id) VALUES (?, ?, ?, ?)",
                     (UID_DOG, "o1.jpg", "p1.jpg", 1))
        conn.execute("INSERT INTO prediction_sessions (uid, original_image, predicted_image, user_id) VALUES (?, ?, ?, ?)",
                     (UID_CAT, "o2.jpg", "p2.jpg", 1))

        # Insert detection objects with known YOLO labels
        conn.execute("INSERT INTO detection_objects (prediction_uid, label, score, box) VALUES (?, ?, ?, ?)",
                     (UID_DOG, "dog", 0.9, "[0,0,1,1]"))
        conn.execute("INSERT INTO detection_objects (prediction_uid, label, score, box) VALUES (?, ?, ?, ?)",
                     (UID_CAT, "cat", 0.88, "[0,0,1,1]"))

    yield

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM detection_objects WHERE prediction_uid IN (?, ?)", (UID_DOG, UID_CAT))
        conn.execute("DELETE FROM prediction_sessions WHERE uid IN (?, ?)", (UID_DOG, UID_CAT))


def test_get_predictions_by_label_dog():
    response = client.get("/predictions/label/dog", auth=("user", "pass"))
    assert response.status_code == 200
    uids = [item["uid"] for item in response.json()]
    assert UID_DOG in uids
    assert UID_CAT not in uids


def test_get_predictions_by_label_cat():
    response = client.get("/predictions/label/cat", auth=("user", "pass"))
    assert response.status_code == 200
    uids = [item["uid"] for item in response.json()]
    assert UID_CAT in uids
    assert UID_DOG not in uids


def test_get_predictions_by_label_invalid():
    response = client.get("/predictions/label/not_a_label", auth=("user", "pass"))
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image type"
