import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app import app
from db import get_db
from app import prediction_count

client = TestClient(app)


class TestPredictionCount(unittest.TestCase):
    def setUp(self):
        # Override dependencies
        app.dependency_overrides[get_db] = lambda: MagicMock()
        app.dependency_overrides[prediction_count.__globals__["get_current_user"]] = lambda: 1

    def tearDown(self):
        app.dependency_overrides = {}

    @patch("app.queries.count_recent_predictions", return_value=0)
    def test_prediction_count_empty(self, mock_count):
        response = client.get("/prediction/count", auth=("user", "pass"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("count"), 0)
        mock_count.assert_called_once()

    @patch("app.queries.count_recent_predictions", return_value=1)
    def test_prediction_count_after_prediction(self, mock_count):
        response = client.get("/prediction/count", auth=("user", "pass"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("count"), 1)
        mock_count.assert_called_once()
