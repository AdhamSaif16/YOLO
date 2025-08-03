import unittest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
import io
import base64
import numpy as np
from app import app
from db import get_db
from app import predict

client = TestClient(app)


def encode_credentials(username, password):
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


class TestPredictEndpoint(unittest.TestCase):
    def setUp(self):
        # Fake image in memory
        self.image = Image.new("RGB", (100, 100), color="blue")
        self.image_bytes = io.BytesIO()
        self.image.save(self.image_bytes, format="JPEG")
        self.image_bytes.seek(0)

        # Mock user and DB
        app.dependency_overrides[get_db] = lambda: MagicMock()
        app.dependency_overrides[predict.__globals__["optional_auth"]] = lambda: None

    def tearDown(self):
        app.dependency_overrides = {}

    def test_predict_no_auth(self):
        mock_result = MagicMock()
        mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_result.boxes = []

        mock_model = MagicMock()
        mock_model.return_value = [mock_result]
        mock_model.names = ["person"]

        with patch("app.model", mock_model):
            response = client.post(
                "/predict",
                files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
            )

        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("prediction_uid", json_data)
        self.assertIn("detection_count", json_data)
        self.assertIn("labels", json_data)
        self.assertIn("time_took", json_data)


    @patch("app.model")
    @patch("app.queries.save_prediction_session")
    def test_predict_with_valid_auth(self, mock_save_session, mock_model):
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_model.return_value = [mock_result]
        mock_model.names = ["person"]

        # Override auth to return user_id
        app.dependency_overrides[predict.__globals__["optional_auth"]] = lambda: MagicMock(username="user", password="pass")
        with patch("app.verify_credentials", return_value=1):
            response = client.post(
                "/predict",
                headers=encode_credentials("user", "pass"),
                files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn("prediction_uid", response.json())
        mock_save_session.assert_called_once()

    @patch("app.model")
    @patch("app.queries.save_prediction_session")
    def test_predict_with_invalid_auth(self, mock_save_session, mock_model):
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_model.return_value = [mock_result]

        # Override to simulate bad auth
        app.dependency_overrides[predict.__globals__["optional_auth"]] = lambda: MagicMock(username="wrong", password="wrong")
        with patch("app.verify_credentials", return_value=None):
            response = client.post(
                "/predict",
                headers=encode_credentials("wrong", "wrong"),
                files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
            )
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json().get("user_id", None))  # user_id not exposed

    def test_predict_missing_file(self):
        response = client.post("/predict", files={})
        self.assertEqual(response.status_code, 422)

    @patch("app.model")
    @patch("app.queries.save_prediction_session")
    @patch("app.queries.save_detection_object")
    def test_predict_with_detection_boxes(self, mock_save_obj, mock_save_session, mock_model):
        mock_box = MagicMock()
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].item.return_value = 0
        mock_box.conf = [MagicMock()]
        mock_box.conf[0].__float__.return_value = 0.9
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist.return_value = [0, 0, 100, 100]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        mock_model.return_value = [mock_result]
        mock_model.names = ["person"]

        app.dependency_overrides[predict.__globals__["optional_auth"]] = lambda: MagicMock(username="user", password="pass")
        with patch("app.verify_credentials", return_value=1):
            response = client.post(
                "/predict",
                headers=encode_credentials("user", "pass"),
                files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
            )

        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["detection_count"], 1)
        self.assertEqual(json_data["labels"], ["person"])
        
    def test_optional_auth_returns_none_on_invalid_credentials(self):
        """
        This test ensures the optional_auth helper returns None
        when authentication is missing or invalid.
        """
        # Remove override to test the actual function
        if predict.__globals__["optional_auth"] in app.dependency_overrides:
            del app.dependency_overrides[predict.__globals__["optional_auth"]]

        with patch("app.verify_credentials", return_value=None), \
            patch("app.model") as mock_model:

            mock_result = MagicMock()
            mock_result.boxes = []
            mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_model.return_value = [mock_result]
            mock_model.names = ["person"]

            response = client.post(
                "/predict",
                files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
                # No headers → triggers optional_auth without credentials
            )

        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("prediction_uid", json_data)
    @patch("app.model")
    @patch("app.queries.save_prediction_session")
    def test_predict_with_http_exception_in_credentials(self, mock_save_session, mock_model):
        """
        This test covers the code path where verify_credentials raises HTTPException
        inside the /predict endpoint.
        """
        mock_result = MagicMock()
        mock_result.boxes = []
        mock_result.plot.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_model.return_value = [mock_result]
        mock_model.names = ["person"]

        # Provide mocked credentials through dependency override
        app.dependency_overrides[predict.__globals__["optional_auth"]] = lambda: MagicMock(username="bad", password="bad")

        # Simulate that verify_credentials raises HTTPException
        with patch("app.verify_credentials", side_effect=HTTPException(status_code=401, detail="Invalid credentials")):
            response = client.post(
                "/predict",
                headers=encode_credentials("bad", "bad"),
                files={"file": ("test.jpg", self.image_bytes, "image/jpeg")}
            )

        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("prediction_uid", json_data)
        self.assertIn("labels", json_data)

        
