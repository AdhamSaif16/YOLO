import unittest
from fastapi.testclient import TestClient
from app import app, init_db  # removed add_test_user

client = TestClient(app)


class TestPredictionImage(unittest.TestCase):
    def setUp(self):
        init_db()  # this is enough if DB already has the user

    def test_get_prediction_image_not_found(self):
        response = client.get(
            "/prediction/nonexistent/image",
            headers={"accept": "image/png"},
            auth=("user", "pass")
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Prediction not found")

    def test_get_prediction_image_unauthorized(self):
        response = client.get(
            "/prediction/nonexistent/image",
            headers={"accept": "image/png"}
            # No auth provided
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")
