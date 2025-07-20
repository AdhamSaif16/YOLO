import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io
import os
import base64

from app import app, DB_PATH, init_db, add_test_user

def auth_headers(username="user", password="pass"):
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

class TestPredictionCount(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()
        add_test_user()

        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

    def test_prediction_count_empty(self):
        response = self.client.get("/prediction/count", headers=auth_headers())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("count"), 0)

    def test_prediction_count_after_prediction(self):
        self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        response = self.client.get("/prediction/count", headers=auth_headers())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("count"), 1)
