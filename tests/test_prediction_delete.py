import sqlite3
import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io
import os
import base64
from app import app, init_db, DB_PATH, add_test_user, save_prediction_session

TEST_UID = "delete-test-uid"
ORIGINAL_PATH = f"uploads/original/{TEST_UID}.jpg"
PREDICTED_PATH = f"uploads/predicted/{TEST_UID}.jpg"

def auth_headers(username="user", password="pass"):
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


class TestDelete(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

        # Fresh DB and folders
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()
        add_test_user()

        os.makedirs("uploads/original", exist_ok=True)
        os.makedirs("uploads/predicted", exist_ok=True)

        # Setup test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

        # Setup dummy files
        with open(ORIGINAL_PATH, "wb") as f:
            f.write(b"test image")
        with open(PREDICTED_PATH, "wb") as f:
            f.write(b"test image")

        save_prediction_session(TEST_UID, ORIGINAL_PATH, PREDICTED_PATH, user_id=1)

    def test_delete_not_authenticated(self):
        response = self.client.delete("/prediction/-1")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")

    def test_delete_prediction_success(self):
        # Make a prediction first
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")},
            headers=auth_headers()
        )
        self.assertEqual(response.status_code, 200)
        uid = response.json()["prediction_uid"]

        response2 = self.client.delete(f"/prediction/{uid}", headers=auth_headers())
        self.assertEqual(response2.status_code, 200)
        self.assertIn("detail", response2.json())

    def test_delete_prediction_files_missing(self):
        # Insert record but remove files
        uid = "no-files-uid"
        orig = f"uploads/original/{uid}.jpg"
        pred = f"uploads/predicted/{uid}.jpg"
        save_prediction_session(uid, orig, pred, user_id=1)

        # No files created
        response = self.client.delete(f"/prediction/{uid}", headers=auth_headers())
        self.assertEqual(response.status_code, 200)
        self.assertIn("no image files", response.json()["detail"])

    def test_delete_prediction_not_found(self):
        response = self.client.delete("/prediction/nonexistent", headers=auth_headers())
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found")

    def test_delete_prediction_not_owned(self):
        # Add second user manually
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO users (username, pass) VALUES (?, ?)", ("another", "pass"))

        # Insert prediction with user_id=2
        uid = "other-user-pred"
        save_prediction_session(uid, "x.jpg", "y.jpg", user_id=2)

        response = self.client.delete(f"/prediction/{uid}", headers=auth_headers())
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Not authorized to delete this prediction")
