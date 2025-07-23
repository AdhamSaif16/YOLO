import unittest
import os
import sqlite3
import base64
from fastapi.testclient import TestClient
from app import app, init_db, DB_PATH, add_test_user, save_prediction_session

UID_OWNED = "get-owned-uid"
UID_OTHER = "get-other-uid"
ORIGINAL_PATH = f"uploads/original/{UID_OWNED}.jpg"
PREDICTED_PATH = f"uploads/predicted/{UID_OWNED}.jpg"

def auth_headers(username="user", password="pass"):
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

class TestGetPredictionByUID(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

        # Reset DB and folders
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

        init_db()
        add_test_user()  # inserts user:pass with id=1

        os.makedirs("uploads/original", exist_ok=True)
        os.makedirs("uploads/predicted", exist_ok=True)

        # Create dummy files
        with open(ORIGINAL_PATH, "wb") as f:
            f.write(b"original image")
        with open(PREDICTED_PATH, "wb") as f:
            f.write(b"predicted image")

        # Prediction owned by user 1
        save_prediction_session(UID_OWNED, ORIGINAL_PATH, PREDICTED_PATH, user_id=1)

        # Prediction owned by another user (user_id = 2)
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO users (username, pass) VALUES (?, ?)", ("user2", "pass2"))
        save_prediction_session(UID_OTHER, "fake1.jpg", "fake2.jpg", user_id=2)

    def test_get_prediction_success(self):
        response = self.client.get(f"/prediction/{UID_OWNED}", headers=auth_headers())
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["uid"], UID_OWNED)
        self.assertIn("timestamp", data)
        self.assertIn("original_image", data)
        self.assertIn("predicted_image", data)

    def test_get_prediction_not_authenticated(self):
        response = self.client.get(f"/prediction/{UID_OWNED}")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")

    def test_get_prediction_not_found(self):
        response = self.client.get("/prediction/nonexistent-uid", headers=auth_headers())
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found")

    def test_get_prediction_not_authorized(self):
        headers = auth_headers("user", "pass")  # logged in as user 1
        response = self.client.get(f"/prediction/{UID_OTHER}", headers=headers)
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Not authorized to access this prediction")
