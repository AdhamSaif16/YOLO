import os
import sqlite3
import pytest
from fastapi.testclient import TestClient
from app import app, DB_PATH, init_db, add_test_user
from PIL import Image

client = TestClient(app)

UID = "test-image-uid"
PREDICTED_DIR = "uploads/predicted"
IMAGE_PATH = os.path.join(PREDICTED_DIR, UID + ".png")  # using PNG now


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup
    init_db()
    add_test_user()
    os.makedirs(PREDICTED_DIR, exist_ok=True)

    # Create a real PNG image
    img = Image.new("RGB", (10, 10), color="red")
    img.save(IMAGE_PATH, "PNG")

    # Insert into DB
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO prediction_sessions (uid, original_image, predicted_image, user_id) VALUES (?, ?, ?, ?)",
                     (UID, "dummy_original.jpg", IMAGE_PATH, 1))

    yield

    # Teardown
    if os.path.exists(IMAGE_PATH):
        os.remove(IMAGE_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (UID,))


def test_get_prediction_image_png_success():
    response = client.get(f"/prediction/{UID}/image", headers={"accept": "image/png"}, auth=("user", "pass"))
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_get_prediction_image_accept_header_does_not_match_actual_format():
    response = client.get(f"/prediction/{UID}/image", headers={"accept": "image/jpeg"}, auth=("user", "pass"))
    assert response.status_code == 200
    assert "image/jpeg" in response.headers["content-type"]


def test_get_prediction_image_not_found():
    response = client.get("/prediction/nonexistent/image", headers={"accept": "image/png"}, auth=("user", "pass"))
    assert response.status_code == 404
    assert response.json()["detail"] == "Prediction not found"


def test_get_prediction_image_file_missing():
    missing_uid = "missing-file-uid"
    fake_path = os.path.join(PREDICTED_DIR, "missing.png")

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO prediction_sessions (uid, original_image, predicted_image, user_id) VALUES (?, ?, ?, ?)",
                     (missing_uid, "xx", fake_path, 1))

    response = client.get(f"/prediction/{missing_uid}/image", headers={"accept": "image/png"}, auth=("user", "pass"))
    assert response.status_code == 404
    assert "Predicted image file not found" in response.text

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (missing_uid,))
