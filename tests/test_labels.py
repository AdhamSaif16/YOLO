import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app import app
from app import get_all_labels
from db import get_db

client = TestClient(app)

class TestGetLabels(unittest.TestCase):
    def setUp(self):
        # Mock DB and Auth
        app.dependency_overrides[get_db] = lambda: MagicMock()
        app.dependency_overrides[get_all_labels.__globals__["get_current_user"]] = lambda: 1

    def tearDown(self):
        app.dependency_overrides = {}

    @patch("app.queries.get_all_labels")
    def test_get_labels_success(self, mock_get_labels):
        mock_get_labels.return_value = [
            {"label": "dog", "count": 1},
            {"label": "cat", "count": 1}
        ]
        response = client.get("/prediction/labels", auth=("user", "pass"))
        self.assertEqual(response.status_code, 200)
        labels = [entry["label"] for entry in response.json()]
        self.assertIn("dog", labels)
        self.assertIn("cat", labels)

    @patch("app.queries.get_all_labels")
    def test_get_labels_empty(self, mock_get_labels):
        mock_get_labels.return_value = []
        response = client.get("/prediction/labels", auth=("user", "pass"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    def test_get_labels_unauthenticated(self):
        # Remove override temporarily
        app.dependency_overrides.pop(get_all_labels.__globals__["get_current_user"], None)

        response = client.get("/prediction/labels")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")

        # Restore override
        app.dependency_overrides[get_all_labels.__globals__["get_current_user"]] = lambda: 1
