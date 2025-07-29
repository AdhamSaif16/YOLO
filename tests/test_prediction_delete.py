import unittest
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from app import app, get_db

client = TestClient(app)

# Mock user credentials
mock_credentials = ("user", "pass")
mock_auth_header = {
    "Authorization": "Basic dXNlcjpwYXNz"  # base64 of 'user:pass'
}


class TestDeletePrediction(unittest.TestCase):
    def setUp(self):
        self.uid = "mocked-uid"
        self.original_path = f"uploads/original/{self.uid}.jpg"
        self.predicted_path = f"uploads/predicted/{self.uid}.jpg"

        # Override get_db with a mock
        self.mock_db = Mock()
        app.dependency_overrides[get_db] = lambda: self.mock_db

    def tearDown(self):
        app.dependency_overrides = {}

    @patch("app.queries.get_user_by_credentials")
    def test_delete_prediction_not_found(self, mock_user):
        self.mock_db = Mock()
        mock_user.return_value = Mock(id=1)

        with patch("app.queries.get_prediction_by_uid", return_value=None):
            response = client.delete(f"/prediction/{self.uid}", headers=mock_auth_header)
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json(), {"detail": "Prediction not found"})

    @patch("os.path.exists", return_value=True)
    @patch("os.remove")
    @patch("app.queries.delete_prediction_session")
    @patch("app.queries.delete_detection_objects_by_uid")
    @patch("app.queries.get_prediction_by_uid")
    @patch("app.queries.get_user_by_credentials")
    def test_delete_prediction_success(
        self,
        mock_user,
        mock_get_prediction,
        mock_delete_detections,
        mock_delete_session,
        mock_remove,
        mock_exists
    ):
        mock_user.return_value = Mock(id=1)
        mock_prediction = Mock(
            uid=self.uid,
            original_image=self.original_path,
            predicted_image=self.predicted_path,
            user_id=1
        )
        mock_get_prediction.return_value = mock_prediction

        response = client.delete(f"/prediction/{self.uid}", headers=mock_auth_header)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text.strip('"'), "Successfully Deleted")

        mock_delete_detections.assert_called_once_with(self.mock_db, self.uid)
        mock_delete_session.assert_called_once_with(self.mock_db, self.uid)
        mock_remove.assert_any_call(self.original_path)
        mock_remove.assert_any_call(self.predicted_path)

    @patch("app.queries.get_user_by_credentials")
    @patch("app.queries.get_prediction_by_uid")
    def test_delete_prediction_unauthorized(self, mock_get_prediction, mock_user):
        mock_user.return_value = Mock(id=1)

        mock_prediction = Mock()
        mock_prediction.uid = self.uid
        mock_prediction.user_id = 999  # not the same user
        mock_prediction.original_image = self.original_path
        mock_prediction.predicted_image = self.predicted_path
        mock_get_prediction.return_value = mock_prediction

        response = client.delete(f"/prediction/{self.uid}", headers=mock_auth_header)

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json(), {"detail": "Unauthorized access to this prediction"})

