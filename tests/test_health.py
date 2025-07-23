import unittest
from fastapi.testclient import TestClient
from app import app

class TestHealthEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_ok(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
        self.assertEqual(response.headers["content-type"], "application/json")
