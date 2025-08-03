import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app import app, verify_credentials, get_current_user, security
from db import get_db


client = TestClient(app)


class FakeCred:
    def __init__(self, username, password):
        self.username = username
        self.password = password


class TestHelperFunctions(unittest.TestCase):
    def setUp(self):
        # Override DB dependency
        app.dependency_overrides[get_db] = lambda: MagicMock()

    def tearDown(self):
        app.dependency_overrides = {}

    @patch("app.queries.get_user_by_credentials")
    def test_verify_credentials_valid(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = 42
        mock_get_user.return_value = mock_user

        cred = FakeCred("user", "pass")
        result = verify_credentials(cred, db=MagicMock())
        self.assertEqual(result, 42)

    @patch("app.queries.get_user_by_credentials", return_value=None)
    def test_verify_credentials_invalid_user(self, mock_get_user):
        cred = FakeCred("wrong", "wrong")
        result = verify_credentials(cred, db=MagicMock())
        self.assertIsNone(result)

    @patch("app.queries.get_user_by_credentials")
    def test_get_current_user_valid(self, mock_get_user):
        mock_user = MagicMock()
        mock_user.id = 1
        mock_get_user.return_value = mock_user

        # Override credentials and db
        app.dependency_overrides[security] = lambda: FakeCred("user", "pass")

        user_id = get_current_user(
            credentials=FakeCred("user", "pass"),
            db=MagicMock()
        )
        self.assertEqual(user_id, 1)

    @patch("app.queries.get_user_by_credentials", return_value=None)
    def test_get_current_user_invalid(self, mock_get_user):
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            get_current_user(
                credentials=FakeCred("wrong", "wrong"),
                db=MagicMock()
            )
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.detail, "Invalid or missing credentials")
