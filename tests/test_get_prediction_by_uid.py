import unittest
import os
import sqlite3
from fastapi.testclient import TestClient
from app import app, init_db, DB_PATH, add_test_user, save_prediction_session
import base64

UID = "test-get-uid"
ORIGINAL_PATH = f"uploads/original/{UID}.jpg"
PREDICTED_PATH = f"uploads/predicted/{UID}.jpg"

def auth_headers(username="user", password="pass"):
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

class TestGetPredictionByUID(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

        init_db()
        add_test_user()

        os.makedirs("uploads/original", exist_ok=True)
        os.makedirs("uploads/predicted", exist_ok=True)

        # Create dummy files
        with open(ORIGINAL_PATH, "wb") as f:
            f.write(b"original image")
        with open(PREDICTED_PATH, "wb") as f:
            f.write(b"predicted image")

        save_prediction_session(UID, ORIGINAL_PATH, PREDICTED_PATH, user_id=1)

    def test_get_prediction_success(self):
        response = self.client.get(f"/prediction/{UID}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["uid"], UID)
        self.assertIn("original_image", data)
        self.assertIn("predicted_image", data)

    def test_get_prediction_not_found(self):
        response = self.client.get("/prediction/nonexistent-uid")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found")

    def test_get_prediction_returns_expected_fields(self):
        response = self.client.get(f"/prediction/{UID}")
        data = response.json()
        self.assertIn("uid", data)
        self.assertIn("timestamp", data)
        self.assertIn("original_image", data)
        self.assertIn("predicted_image", data)

    def test_get_prediction_returns_json(self):
        response = self.client.get(f"/prediction/{UID}")
        self.assertEqual(response.headers["content-type"], "application/json")
