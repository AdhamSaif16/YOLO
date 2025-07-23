import unittest
from fastapi.testclient import TestClient
from PIL import Image
import io
import os
import sqlite3
import base64
import numpy as np
from unittest.mock import patch, MagicMock
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

        self.image = Image.new("RGB", (100, 100), color="blue")
        self.image_bytes = io.BytesIO()
        self.image.save(self.image_bytes, format="JPEG")
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
        self.image_bytes.seek(0)
        response = self.client.post(
            "/predict",
            headers=encode_credentials("user", "pass"),
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("prediction_uid", json_data)
        self.assertIn("labels", json_data)

        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT user_id FROM prediction_sessions WHERE uid = ?", (json_data["prediction_uid"],)).fetchone()
            self.assertIsNotNone(row)
            self.assertIsNotNone(row[0])

    def test_predict_with_invalid_auth(self):
        self.image_bytes.seek(0)
        response = self.client.post(
            "/predict",
            headers=encode_credentials("wrong", "wrong"),
            files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
        )
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute("SELECT user_id FROM prediction_sessions WHERE uid = ?", (json_data["prediction_uid"],)).fetchone()
            self.assertIsNone(row[0])

    def test_predict_missing_file(self):
        response = self.client.post("/predict", files={})
        self.assertEqual(response.status_code, 422)

    def test_predict_with_detection_boxes(self):
        self.image_bytes.seek(0)

        # Mock a box with .cls, .conf, .xyxy and .tolist()
        mock_box = MagicMock()
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].item.return_value = 0

        mock_box.conf = [MagicMock()]
        mock_box.conf[0] = MagicMock()
        mock_box.conf[0].__float__.return_value = 0.9

        mock_xyxy_entry = MagicMock()
        mock_xyxy_entry.tolist.return_value = [0, 0, 100, 100]
        mock_box.xyxy = [mock_xyxy_entry]

        # Mock the result and model
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_model.names = ["person"]

        with patch("app.model", mock_model):
            response = self.client.post(
                "/predict",
                headers=encode_credentials("user", "pass"),
                files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
            )

            self.assertEqual(response.status_code, 200)
            json_data = response.json()
            self.assertEqual(json_data["detection_count"], 1)
            self.assertEqual(json_data["labels"], ["person"])
