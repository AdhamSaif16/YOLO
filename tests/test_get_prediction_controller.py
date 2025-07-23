import os
import unittest
import sqlite3
import base64
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app import (
    app, DB_PATH, init_db,
    add_test_user, verify_credentials
)

def encode_credentials(username, password):
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

class TestHelperFunctions(unittest.TestCase):
    def setUp(self):
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        init_db()
        add_test_user()
        self.client = TestClient(app)

    def test_add_test_user_creates_user(self):
        with sqlite3.connect(DB_PATH) as conn:
            result = conn.execute("SELECT * FROM users WHERE username = ?", ("user",)).fetchone()
            self.assertIsNotNone(result)
            self.assertEqual(result[1], "user")
            self.assertEqual(result[2], "pass")

    def test_verify_credentials_valid(self):
        credentials = type("FakeCred", (), {
            "username": "user", "password": "pass"
        })()
        user_id = verify_credentials(credentials)
        self.assertIsInstance(user_id, int)

    def test_verify_credentials_invalid_user(self):
        credentials = type("FakeCred", (), {
            "username": "wrong", "password": "wrong"
        })()
        user_id = verify_credentials(credentials)
        self.assertIsNone(user_id)

    def test_verify_credentials_none(self):
        self.assertIsNone(verify_credentials(None))

    def test_get_current_user_valid(self):
        # Access a protected endpoint using valid credentials
        response = self.client.get("/labels", headers=encode_credentials("user", "pass"))
        self.assertEqual(response.status_code, 200)

    def test_get_current_user_invalid(self):
        response = self.client.get("/labels", headers=encode_credentials("wrong", "wrong"))
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Invalid or missing credentials")

    def test_get_current_user_missing(self):
        response = self.client.get("/labels")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Not authenticated")

