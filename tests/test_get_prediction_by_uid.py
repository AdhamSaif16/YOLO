import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app import app, get_prediction
from db import get_db


client = TestClient(app)

class FakePrediction:
    def __init__(self, uid, timestamp, original_image, predicted_image, user_id):
        self.uid = uid
        self.timestamp = timestamp
        self.original_image = original_image
        self.predicted_image = predicted_image
        self.user_id = user_id


class TestGetPredictionByUID(unittest.TestCase):
    def setUp(self):
        def override_get_db():
            return MagicMock()

        def override_get_current_user():
            return 1  # logged in user_id

        app.dependency_overrides[get_db] = override_get_db
        app.dependency_overrides[get_prediction.__globals__["get_current_user"]] = override_get_current_user

    def tearDown(self):
        app.dependency_overrides = {}

    @patch("app.queries.get_prediction_by_uid")
    def test_get_prediction_success(self, mock_query):
        mock_query.return_value = FakePrediction(
            uid="get-owned-uid",
            timestamp="2023-01-01T10:00:00",
            original_image="original.jpg",
            predicted_image="predicted.jpg",
            user_id=1
        )

        response = client.get("/prediction/get-owned-uid", auth=("user", "pass"))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["uid"], "get-owned-uid")
        self.assertIn("timestamp", data)
        self.assertIn("original_image", data)
        self.assertIn("predicted_image", data)

    @patch("app.queries.get_prediction_by_uid")
    def test_get_prediction_not_found(self, mock_query):
        mock_query.return_value = None
        response = client.get("/prediction/nonexistent-uid", auth=("user", "pass"))
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found")

    @patch("app.queries.get_prediction_by_uid")
    def test_get_prediction_not_authorized(self, mock_query):
        # Simulate prediction owned by a different user (id=2)
        mock_query.return_value = FakePrediction(
            uid="get-other-uid",
            timestamp="2023-01-01T11:00:00",
            original_image="o2.jpg",
            predicted_image="p2.jpg",
            user_id=2
        )

        response = client.get("/prediction/get-other-uid", auth=("user", "pass"))
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Unauthorized access to this prediction")

    def test_get_prediction_not_authenticated(self):
        # Temporarily remove override for this test
        app.dependency_overrides.pop(get_prediction.__globals__["get_current_user"], None)
        
        response = client.get("/prediction/get-owned-uid")
        
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")

        # Restore it so it doesn’t affect other tests
        app.dependency_overrides[get_prediction.__globals__["get_current_user"]] = lambda: 1

