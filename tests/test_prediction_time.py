import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io
import os
import base64
from app import app, init_db, DB_PATH, add_test_user

def auth_headers(username="user", password="pass"):
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

class TestDelete(unittest.TestCase):

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

    def test_delete_not_authenticated(self):
        response = self.client.delete("/prediction/-1")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")

    def test_delete_prediction(self):
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        uid = response.json()["prediction_uid"]

        # Send DELETE request with authentication
        response2 = self.client.delete(f"/prediction/{uid}", headers=auth_headers())

        # Handle cases where the predicted image had no objects
        if response.json()["detection_count"] == 0:
            self.assertEqual(response2.status_code, 200)
        else:
            self.assertEqual(response2.status_code, 200)
            self.assertEqual(response2.json(), "Successfully Deleted")
