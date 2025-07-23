import unittest
import os
import sqlite3
import base64
from fastapi.testclient import TestClient
from app import app, DB_PATH, init_db, add_test_user, save_prediction_session, save_detection_object

UID_1 = "labels-uid-1"
UID_2 = "labels-uid-2"
LABELS = ["dog", "cat"]

def auth_headers(username="user", password="pass"):
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

class TestGetLabels(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

        # Reset DB
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()
        add_test_user()

        # Insert two predictions and labels
        save_prediction_session(UID_1, "a.jpg", "a_pred.jpg", user_id=1)
        save_detection_object(UID_1, "dog", 0.95, "[0,0,1,1]")

        save_prediction_session(UID_2, "b.jpg", "b_pred.jpg", user_id=1)
        save_detection_object(UID_2, "cat", 0.88, "[2,2,3,3]")

    def test_get_labels_success(self):
        response = self.client.get("/labels", headers=auth_headers())
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("labels", data)
        self.assertIsInstance(data["labels"], list)
        self.assertIn("dog", data["labels"])
        self.assertIn("cat", data["labels"])

    def test_get_labels_empty(self):
        # Clear labels
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM detection_objects")
            conn.execute("DELETE FROM prediction_sessions")

        response = self.client.get("/labels", headers=auth_headers())
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["labels"], [])

    def test_get_labels_unauthenticated(self):
        response = self.client.get("/labels")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")
