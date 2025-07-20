import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io
import os
from app import app, init_db,DB_PATH


class Test_Delete(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)

        init_db()
        
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

    def test_detete_not_found(self):
        response2 = self.client.delete("/prediction/-1")
        self.assertEqual(response2.status_code, 401)
        self.assertEqual(response2.json()["detail"], "Prediction not found")

    
    def test_delete_prediction(self):
        
        response = self.client.post(
            "/predict",
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        print(response.json())
        uid = response.json()["prediction_uid"]

        if response.json()["detection_count"] == 0:
            self.assertEqual(self.client.delete(f"/prediction/{uid}").status_code, 404)
        else:
            response2 = self.client.delete(f"/prediction/{uid}")
            self.assertEqual(response2.status_code, 200)
            self.assertEqual(response2.json()["message"], "Successfully Deleted")