import os
import sqlite3
import pytest
from fastapi.testclient import TestClient
from app import app, DB_PATH, init_db, add_test_user

client = TestClient(app)

UID = "image-test-uid"
FILENAME = UID + ".jpg"
ORIGINAL_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
ORIGINAL_PATH = os.path.join(ORIGINAL_DIR, FILENAME)
PREDICTED_PATH = os.path.join(PREDICTED_DIR, FILENAME)


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    init_db()
    add_test_user()
    os.makedirs(ORIGINAL_DIR, exist_ok=True)
    os.makedirs(PREDICTED_DIR, exist_ok=True)

    with open(ORIGINAL_PATH, "wb") as f:
        f.write(b"\xFF\xD8\xFF\xE0" + b"\x00" * 100)  # minimal JPEG header

    with open(PREDICTED_PATH, "wb") as f:
        f.write(b"\xFF\xD8\xFF\xE0" + b"\x00" * 100)

    yield

    if os.path.exists(ORIGINAL_PATH):
        os.remove(ORIGINAL_PATH)
    if os.path.exists(PREDICTED_PATH):
        os.remove(PREDICTED_PATH)


def test_get_original_image_success():
    response = client.get(f"/image/original/{FILENAME}", auth=("user", "pass"))
    assert response.status_code == 200
    assert response.headers["content-type"] in ("image/jpeg", "application/octet-stream")


def test_get_predicted_image_success():
    response = client.get(f"/image/predicted/{FILENAME}", auth=("user", "pass"))
    assert response.status_code == 200
    assert response.headers["content-type"] in ("image/jpeg", "application/octet-stream")


def test_get_image_file_not_found():
    response = client.get("/image/original/missing-file.jpg", auth=("user", "pass"))
    assert response.status_code == 404
    assert response.json()["detail"] == "Image not found"


def test_get_image_invalid_type():
    response = client.get(f"/image/invalid_type/{FILENAME}", auth=("user", "pass"))
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image type"
