import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app import app, get_predictions_by_label
from db import get_db

client = TestClient(app)

class TestGetPredictionsByLabel(unittest.TestCase):
    def setUp(self):
        def override_get_current_user():
            return 1

        def override_get_db():
            return MagicMock()

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_predictions_by_label.__globals__["get_current_user"]] = override_get_current_user

    def tearDown(self):
        app.dependency_overrides = {}

    @patch("app.queries.get_predictions_by_label")
    def test_get_predictions_by_label_dog(self, mock_query):
        mock_query.return_value = [{"uid": "label-dog-1", "timestamp": "2024-01-01T10:00:00"}]

        response = client.get("/predictions/label/dog", auth=("user", "pass"))
        self.assertEqual(response.status_code, 200)
        uids = [item["uid"] for item in response.json()]
        self.assertIn("label-dog-1", uids)
        self.assertNotIn("label-cat-1", uids)

    @patch("app.queries.get_predictions_by_label")
    def test_get_predictions_by_label_cat(self, mock_query):
        mock_query.return_value = [{"uid": "label-cat-1", "timestamp": "2024-01-01T11:00:00"}]

        response = client.get("/predictions/label/cat", auth=("user", "pass"))
        self.assertEqual(response.status_code, 200)
        uids = [item["uid"] for item in response.json()]
        self.assertIn("label-cat-1", uids)
        self.assertNotIn("label-dog-1", uids)

    def test_get_predictions_by_label_invalid(self):
        response = client.get("/predictions/label/not_a_label", auth=("user", "pass"))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Invalid label")
