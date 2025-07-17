import glob
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from ultralytics import YOLO
from PIL import Image
import sqlite3
import os
import uuid
import shutil
from typing import Optional
from fastapi import Depends
from starlette.status import HTTP_401_UNAUTHORIZED

# Disable GPU usage
import torch
torch.cuda.is_available = lambda: False

app = FastAPI()

security = HTTPBasic()

UPLOAD_DIR = "uploads/original"
PREDICTED_DIR = "uploads/predicted"
DB_PATH = "predictions.db"
labels = [
   "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PREDICTED_DIR, exist_ok=True)

# Download the AI model (tiny model ~6MB)
model = YOLO("yolov8n.pt")  

############################################################### helper functions ###############################################

#adding one demo user for testing: "user:pass"
def add_test_user():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT OR IGNORE INTO users (username, pass) VALUES (?, ?)", ("user", "pass"))

#a function that verifies that the credintials are right
#return:
#       id: if exist 
#       none: else
# the optional is to for the optional predict endpoint
def verify_credentials(credentials: Optional[HTTPBasicCredentials]) -> Optional[int]:
    """
    Verify provided credentials. Return user_id if valid, None otherwise.
    """
    if credentials is None:
        return None

    username = credentials.username
    password = credentials.password

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        user = conn.execute("SELECT * FROM users WHERE username = ? AND pass = ?", (username, password)).fetchone()
        return user["id"] if user else None

def get_current_user(credentials: Optional[HTTPBasicCredentials] = Depends(security)) -> Optional[int]:
    user_id = verify_credentials(credentials)
    if user_id is None:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return user_id
##########################################################  end of helper functions ############################################
# Initialize SQLite
def init_db():
    with sqlite3.connect(DB_PATH) as conn:

        # Create users table to store the users credentials
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username UNIQUE NOT NULL,
                pass TEXT
            )
        """)

        # Create the predictions main table to store the prediction session
        conn.execute("""
            CREATE TABLE IF NOT EXISTS prediction_sessions (
                uid TEXT PRIMARY KEY,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                original_image TEXT,
                predicted_image TEXT,
                user_id INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Create the objects table to store individual detected objects in a given image
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_objects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_uid TEXT,
                label TEXT,
                score REAL,
                box TEXT,
                FOREIGN KEY (prediction_uid) REFERENCES prediction_sessions (uid)
            )
        """)
        
        # Create index for faster queries
        conn.execute("CREATE INDEX IF NOT EXISTS idx_prediction_uid ON detection_objects (prediction_uid)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_label ON detection_objects (label)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_score ON detection_objects (score)")

init_db()
add_test_user()

def save_prediction_session(uid, original_image, predicted_image,user_id=None):
    """
    Save prediction session to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO prediction_sessions (uid, original_image, predicted_image,user_id)
            VALUES (?, ?, ?, ?)
        """, (uid, original_image, predicted_image,user_id))

def save_detection_object(prediction_uid, label, score, box):
    """
    Save detection object to database
    """
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO detection_objects (prediction_uid, label, score, box)
            VALUES (?, ?, ?, ?)
        """, (prediction_uid, label, score, str(box)))

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """
    Predict objects in an image
    """
    start_time = time.time()
    ext = os.path.splitext(file.filename)[1]
    uid = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)

    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = model(original_path, device="cpu")

    annotated_frame = results[0].plot()  # NumPy image with boxes
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    save_prediction_session(uid, original_path, predicted_path)
    
    detected_labels = []
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        save_detection_object(uid, label, score, bbox)
        detected_labels.append(label)

    time_took = round(time.time() - start_time,2)
    return {
        "prediction_uid": uid, 
        "detection_count": len(results[0].boxes),
        "labels": detected_labels,
        "time_took": time_took
    }
@app.get("/prediction/count")
def get_prediction_count(user_id: int = Depends(get_current_user)):
    """
    Get prediction count from last week
    """
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT count(*) FROM prediction_sessions WHERE timestamp >= DATETIME('now', '-7 days')").fetchall()
    return {"count": count[0][0]} 

@app.get("/labels")
def get_uniqe_labels():
    """
    Get all unique labels from detection objects
    """
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT DISTINCT label FROM detection_objects do join prediction_sessions ps ON do.prediction_uid = ps.uid WHERE ps.timestamp >= DATETIME('now', '-7 days')").fetchall()
    labels=[]
    for row in rows:
        labels.append(row[0])
    return {"labels": labels}

@app.delete("/prediction/{uid}")
def delete_prediction(uid: str):
    with sqlite3.connect(DB_PATH) as conn:
        con1 = conn.execute("DELETE FROM detection_objects WHERE prediction_uid = ?", (uid,))
        if con1.rowcount == 0:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        con2 = conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (uid,))
        if con2.rowcount == 0:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        conn.commit()

    # Check for the file with any of the known image extensions
    deleted = False
    for ext in [".jpg", ".jpeg", ".png"]:
        upload_path = os.path.join(UPLOAD_DIR, uid + ext)
        predict_path = os.path.join(PREDICTED_DIR, uid + ext)

        if os.path.exists(upload_path):
            os.remove(upload_path)
            deleted = True
        if os.path.exists(predict_path):
            os.remove(predict_path)
            deleted = True

    if not deleted:
        raise HTTPException(status_code=404, detail="Prediction file not found")

    return "Successfully Deleted"
        

@app.delete("/prediction/{uid}")
def delete_prediction(uid: str):
    with sqlite3.connect(DB_PATH) as conn:
        con1 = conn.execute("DELETE FROM detection_objects WHERE prediction_uid = ?", (uid,))
        if con1.rowcount == 0:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        con2 = conn.execute("DELETE FROM prediction_sessions WHERE uid = ?", (uid,))
        if con2.rowcount == 0:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        conn.commit()

    deleted = False
    for ext in [".jpg", ".jpeg", ".png"]:
        upload_path = os.path.join(UPLOAD_DIR, uid + ext)
        predict_path = os.path.join(PREDICTED_DIR, uid + ext)

        if os.path.exists(upload_path):
            os.remove(upload_path)
            deleted = True
        if os.path.exists(predict_path):
            os.remove(predict_path)
            deleted = True

    if not deleted:
        raise HTTPException(status_code=404, detail="Prediction file not found")

    return "Successfully Deleted"

@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str):
    """
    Get prediction sessions containing objects with specified label
    """
    if label not in labels:
        raise HTTPException(status_code=400, detail="Invalid image type")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT DISTINCT ps.uid, ps.timestamp
            FROM prediction_sessions ps
            JOIN detection_objects do ON ps.uid = do.prediction_uid
            WHERE do.label = ?
        """, (label,)).fetchall()
        
        return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]

@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(min_score: float):
    """
    Get prediction sessions containing objects with score >= min_score
    """
    if min_score <0 or min_score > 1:
        raise HTTPException(status_code=400, detail="Invalid score")
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
            SELECT DISTINCT ps.uid, ps.timestamp
            FROM prediction_sessions ps
            JOIN detection_objects do ON ps.uid = do.prediction_uid
            WHERE do.score >= ?
        """, (min_score,)).fetchall()
        
        return [{"uid": row["uid"], "timestamp": row["timestamp"]} for row in rows]

@app.get("/image/{type}/{filename}")
def get_image(type: str, filename: str):
    """
    Get image by type and filename
    """
    if type not in ["original", "predicted"]:
        raise HTTPException(status_code=400, detail="Invalid image type")
    path = os.path.join("uploads", type, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)

@app.get("/prediction/{uid}/image")
def get_prediction_image(uid: str, request: Request):
    """
    Get prediction image by uid
    """
    accept = request.headers.get("accept", "")
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("SELECT predicted_image FROM prediction_sessions WHERE uid = ?", (uid,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Prediction not found")
        image_path = row[0]

    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Predicted image file not found")

    if "image/png" in accept:
        return FileResponse(image_path, media_type="image/png")
    elif "image/jpeg" in accept or "image/jpg" in accept:
        return FileResponse(image_path, media_type="image/jpeg")
    else:
        # If the client doesn't accept image, respond with 406 Not Acceptable
        raise HTTPException(status_code=406, detail="Client does not accept an image format") 
    

@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080,reload=True)
