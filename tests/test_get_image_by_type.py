# tests/test_get_image_by_type.py

import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app import app, get_image
from db import get_db


client = TestClient(app)


class FakeResponse:
    def __init__(self):
        self.status_code = 200
    async def __call__(self, scope, receive, send):
        pass


class TestGetImageEndpoint(unittest.TestCase):
    def setUp(self):
        def override_get_current_user():
            return 1

        def override_get_db():
            return MagicMock()

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_image.__globals__["get_current_user"]] = override_get_current_user

    def tearDown(self):
        app.dependency_overrides = {}

    @patch("app.FileResponse", return_value=FakeResponse())
    @patch("os.path.exists", return_value=True)
    def test_get_original_image_success(self, mock_exists, mock_file_response):
        response = client.get("/image/original/test.jpg", auth=("user", "pass"))
        self.assertEqual(response.status_code, 200)
        mock_file_response.assert_called_once()

    @patch("app.FileResponse", return_value=FakeResponse())
    @patch("os.path.exists", return_value=True)
    def test_get_predicted_image_success(self, mock_exists, mock_file_response):
        response = client.get("/image/predicted/test.jpg", auth=("user", "pass"))
        self.assertEqual(response.status_code, 200)
        mock_file_response.assert_called_once()

    def test_get_image_invalid_type(self):
        response = client.get("/image/invalid/test.jpg", auth=("user", "pass"))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid image type")

    @patch("os.path.exists", return_value=False)
    def test_get_image_file_not_found(self, mock_exists):
        response = client.get("/image/original/missing.jpg", auth=("user", "pass"))
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Image not found")
