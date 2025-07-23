import os
import sqlite3
import pytest
from fastapi.testclient import TestClient
from app import app, DB_PATH, init_db, add_test_user

client = TestClient(app)

PREDICTED_DIR = "uploads/predicted"
UID = "test-image-uid"
IMAGE_PATH = os.path.join(PREDICTED_DIR, UID + ".jpg")


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup: create db, user, folders and dummy predicted image
    init_db()
    add_test_user()
    os.makedirs(PREDICTED_DIR, exist_ok=True)

    with open(IMAGE_PATH, "wb") as f:
        f.write(b"\xFF\xD8\xFF\xE0" + b"\x00" * 100)  # minimal JPEG header

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO prediction_sessions (uid, original_image, predicted_image, user_id) VALUES (?, ?, ?, ?)",
                     (UID, "dummy_original.jpg", IMAGE_PATH, 1))

    yield

    # Teardown
    if os.path.exists(IMAGE_PATH):
        os.remove(IMAGE_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (UID,))


def test_get_prediction_image_jpeg_success():
    headers = {"accept": "image/jpeg"}
    response = client.get(f"/prediction/{UID}/image", headers=headers, auth=("user", "pass"))

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xFF\xD8")  # JPEG header


def test_get_prediction_image_not_found():
    response = client.get("/prediction/nonexistent/image", headers={"accept": "image/jpeg"}, auth=("user", "pass"))
    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found"


def test_get_prediction_image_file_missing():
    # Insert session with missing file
    missing_uid = "missing-file-uid"
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO prediction_sessions (uid, original_image, predicted_image, user_id) VALUES (?, ?, ?, ?)",
                     (missing_uid, "xx", "uploads/predicted/missing.jpg", 1))

    response = client.get(f"/prediction/{missing_uid}/image", headers={"accept": "image/jpeg"}, auth=("user", "pass"))
    assert response.status_code == 404
    assert "Predicted image file not found" in response.text

    # Cleanup
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (missing_uid,))


def test_get_prediction_image_not_acceptable():
    response = client.get(f"/prediction/{UID}/image", headers={"accept": "application/json"}, auth=("user", "pass"))
    assert response.status_code == 406
    assert response.json()["detail"] == "Client does not accept an image format"
