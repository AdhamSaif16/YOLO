# tests/test_prediction_uid_image.py

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi.responses import Response
from app import app
from db import get_db

client = TestClient(app)

class FakePrediction:
    def __init__(self, uid, predicted_image, user_id):
        self.uid = uid
        self.predicted_image = predicted_image
        self.user_id = user_id


class TestPredictionImageEndpoint(unittest.TestCase):
    def setUp(self):
        app.dependency_overrides[get_db] = lambda: MagicMock()

    def tearDown(self):
        app.dependency_overrides = {}

    @patch("app.verify_credentials", return_value=1)
    @patch("app.queries.get_prediction_by_uid")
    @patch("app.FileResponse")
    @patch("os.path.exists", return_value=True)
    def test_get_prediction_image_jpeg_success(self, mock_exists, mock_fileresponse, mock_get_pred, mock_verify):
        mock_get_pred.return_value = FakePrediction("abc", "uploads/predicted/abc.jpg", user_id=1)
        mock_fileresponse.return_value = Response(content=b"fake-image-bytes", media_type="image/jpeg")

        response = client.get("/prediction/abc/image", auth=("user", "pass"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"fake-image-bytes")
        self.assertEqual(response.headers["content-type"], "image/jpeg")

    @patch("app.verify_credentials", return_value=1)
    @patch("app.queries.get_prediction_by_uid")
    @patch("os.path.exists", return_value=False)
    def test_get_prediction_image_file_missing(self, mock_exists, mock_get_pred, mock_verify):
        mock_get_pred.return_value = FakePrediction("abc", "uploads/predicted/abc.jpg", user_id=1)

        response = client.get("/prediction/abc/image", auth=("user", "pass"))
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Image not found")
