import datetime
import os
import sqlite3
import unittest
import uuid
from fastapi.testclient import TestClient
from PIL import Image
import io

from app import app, DB_PATH, init_db

class TestUniqueLabels(unittest.TestCase):
    def setUp(self):
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        self.client = TestClient(app)

        init_db()
        # Create a simple test image
        self.test_image = Image.new('RGB', (100, 100), color='red')
        self.image_bytes = io.BytesIO()
        self.test_image.save(self.image_bytes, format='JPEG')
        self.image_bytes.seek(0)

    def test_unique_labels_zero(self):
        response = self.client.get("/labels")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data.get("labels")), 0)



    def test_unique(self):
        uid1 = str(uuid.uuid4())
        uid2 = str(uuid.uuid4())

        now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO prediction_sessions (uid, timestamp) VALUES (?, ?)", (uid1, now))
            conn.execute("INSERT INTO prediction_sessions (uid, timestamp) VALUES (?, ?)", (uid2, now))
            conn.execute("INSERT INTO detection_objects (prediction_uid, label) VALUES (?, ?)", (uid1, "cat"))
            conn.execute("INSERT INTO detection_objects (prediction_uid, label) VALUES (?, ?)", (uid2, "dog"))
            conn.commit()

        response = self.client.get("/labels")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("labels"), ['cat', 'dog'])
    
    def test_not_unique(self):
        uid1 = str(uuid.uuid4())
        uid2 = str(uuid.uuid4())

        now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')

        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO prediction_sessions (uid, timestamp) VALUES (?, ?)", (uid1, now))
            conn.execute("INSERT INTO prediction_sessions (uid, timestamp) VALUES (?, ?)", (uid2, now))
            conn.execute("INSERT INTO detection_objects (prediction_uid, label) VALUES (?, ?)", (uid1, "cat"))
            conn.execute("INSERT INTO detection_objects (prediction_uid, label) VALUES (?, ?)", (uid2, "cat"))
            conn.commit()

        response = self.client.get("/labels")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("labels"), ['cat'])