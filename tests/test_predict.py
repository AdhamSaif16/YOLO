import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io
import os
import sqlite3
import base64

from app import app, DB_PATH, init_db, add_test_user

def encode_credentials(username, password):
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

class TestPredictEndpoint(unittest.TestCase):
    def setUp(self):
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()
        add_test_user()
        self.client = TestClient(app)

        # Create dummy image
        self.image = Image.new('RGB', (100, 100), color='blue')
        self.image_bytes = io.BytesIO()
        self.image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

    def test_predict_no_auth(self):
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("prediction_uid", json_data)
        self.assertIn("detection_count", json_data)
        self.assertIn("labels", json_data)
        self.assertIn("time_took", json_data)


    def test_predict_with_valid_auth(self):
        self.image_bytes.seek(0)  # reset buffer
        response = self.client.post(
            "/predict",
            headers=encode_credentials("user", "pass"),
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("prediction_uid", json_data)
        self.assertIn("labels", json_data)

        # Check DB user_id is saved
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT user_id FROM prediction_sessions WHERE uid = ?", (json_data["prediction_uid"],)).fetchone()
            self.assertIsNotNone(row)
            self.assertIsNone(row[0])

    def test_predict_with_invalid_auth(self):
        self.image_bytes.seek(0)
        response = self.client.post(
            "/predict",
            headers=encode_credentials("wrong", "wrong"),
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        # Should work, but user_id should be NULL in DB
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT user_id FROM prediction_sessions WHERE uid = ?", (json_data["prediction_uid"],)).fetchone()
            self.assertIsNone(row[0])

    def test_predict_missing_file(self):
        response = self.client.post("/predict", files={})
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity due to missing file

