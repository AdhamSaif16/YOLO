import unittest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from app import app
from db import get_db
from app import get_current_user

class FakeDetectionObject:
    def __init__(self, label, score, box, prediction_uid):
        self.label = label
        self.score = score
        self.box = box
        self.prediction_uid = prediction_uid

class TestGetPredictionsByScore(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.credentials = ("user", "pass")

        def override_get_db():
            return Mock()

        app.dependency_overrides[get_db] = override_get_db

    def tearDown(self):
        app.dependency_overrides = {}

    @patch("queries.get_predictions_by_score")
    @patch("app.verify_credentials")
    def test_get_predictions_above_0_5(self, mock_verify_credentials, mock_get_predictions_by_score):
        mock_verify_credentials.return_value = 1
        mock_get_predictions_by_score.return_value = [
            FakeDetectionObject("cat", 0.7, "[10,20,30,40]", "score-high")
        ]

        response = self.client.get(
            "/predictions/score/0.5",
            auth=self.credentials
        )

        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIsInstance(json_data, list)
        self.assertGreater(len(json_data), 0)
        self.assertIn("score-high", [item["prediction_uid"] for item in json_data])

    @patch("queries.get_predictions_by_score")
    @patch("app.verify_credentials")
    def test_get_predictions_empty_result(self, mock_verify_credentials, mock_get_predictions_by_score):
        mock_verify_credentials.return_value = 1
        mock_get_predictions_by_score.return_value = []

        response = self.client.get("/predictions/score/0.9", auth=self.credentials)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])

    @patch("app.verify_credentials")
    def test_prediction_score_missing_credentials(self, mock_verify_credentials):
        mock_verify_credentials.return_value = None

        response = self.client.get("/predictions/score/0.5", auth=self.credentials)

        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"detail": "Invalid or missing credentials"})

    @patch("app.verify_credentials")
    def test_prediction_score_invalid_value(self, mock_verify_credentials):
        mock_verify_credentials.return_value = 1

        response = self.client.get("/predictions/score/-0.1", auth=self.credentials)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Invalid score"})

        response = self.client.get("/predictions/score/1.1", auth=self.credentials)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Invalid score"})


    def test_prediction_score_user_id_none(self):
        # This test overrides get_current_user to return None,
        # so the if user_id is None: line is triggered
        app.dependency_overrides[get_current_user] = lambda: None

        response = self.client.get("/predictions/score/0.5", auth=self.credentials)
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json(), {"detail": "Authentication required"})

        # Clean override
        app.dependency_overrides.pop(get_current_user, None)

    @patch("app.verify_credentials")
    def test_prediction_score_manual_range_check(self, mock_verify_credentials):
        mock_verify_credentials.return_value = 1

        response = self.client.get("/predictions/score/-0.1", auth=self.credentials)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Invalid score"})

        response = self.client.get("/predictions/score/1.1", auth=self.credentials)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Invalid score"})

