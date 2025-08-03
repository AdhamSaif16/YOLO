from datetime import datetime
import glob
import time
from fastapi import FastAPI, Path, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sqlalchemy.orm import Session 
from ultralytics import YOLO
from PIL import Image
import sqlite3
import os
import uuid
import shutil
from typing import Optional
from fastapi import Depends
from starlette.status import HTTP_401_UNAUTHORIZED
from typing import Annotated
from db import engine
from models import Base
from fastapi import Query
import torch
from fastapi.responses import FileResponse
from db import get_db
import queries
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

#a function that verifies that the credintials are right
#return:
#       id: if exist 
#       none: else
# the optional is to for the optional predict endpoint

def verify_credentials(credentials: HTTPBasicCredentials, db: Session) -> int | None:
    user = queries.get_user_by_credentials(db, credentials.username, credentials.password)
    return user.id if user else None


async def optional_auth(request: Request) -> Optional[HTTPBasicCredentials]:
    try:
        return await security(request)
    except HTTPException:
        return None
    
def get_current_user(
    credentials: Optional[HTTPBasicCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> int | None:
    user_id = verify_credentials(credentials, db)
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
    Base.metadata.create_all(bind=engine)

init_db()
db = next(get_db())
queries.add_test_user(db)


@app.post("/predict")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    credentials: Annotated[Optional[HTTPBasicCredentials], Depends(optional_auth)] = None
):
    """
    Predict objects in an image
    """
    db = next(get_db())
    username = None
    if credentials:
        try:
            username = verify_credentials(credentials,db)
        except HTTPException:
            username = None     #Invalid credentials still allow prediction, username remains null

    start_time = time.time()

    ext = os.path.splitext(file.filename)[1]
    uid = str(uuid.uuid4())
    original_path = os.path.join(UPLOAD_DIR, uid + ext)
    predicted_path = os.path.join(PREDICTED_DIR, uid + ext)

    with open(original_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    results = model(original_path, device="cpu")

    annotated_frame = results[0].plot()
    annotated_image = Image.fromarray(annotated_frame)
    annotated_image.save(predicted_path)

    
    queries.save_prediction_session(db,uid, original_path, predicted_path,username)

    detected_labels = []
    for box in results[0].boxes:
        label_idx = int(box.cls[0].item())
        label = model.names[label_idx]
        score = float(box.conf[0])
        bbox = box.xyxy[0].tolist()
        queries.save_detection_object(db, uid, label, score, str(bbox))
        detected_labels.append(label)

    processing_time = round(time.time() - start_time, 2)
    db.close()
    return {
        "prediction_uid": uid,
        "detection_count": len(results[0].boxes),
        "labels": detected_labels,
        "time_took": processing_time
    }

@app.get("/prediction/count")
def prediction_count(db: Session = Depends(get_db), credentials: HTTPBasicCredentials = Depends(security)) -> dict:
    verify_credentials(credentials, db)
    count = queries.count_recent_predictions(db)
    return {"count": count}

@app.get("/prediction/labels")
def get_all_labels(
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    return queries.get_all_labels(db,user_id)


@app.get("/prediction/{uid}")
def get_prediction(
    uid: str,
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    prediction = queries.get_prediction_by_uid(db, uid)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    
    if prediction.user_id != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized access to this prediction")
    
    return {
        "uid": prediction.uid,
        "timestamp": prediction.timestamp,
        "original_image": prediction.original_image,
        "predicted_image": prediction.predicted_image
    }


@app.get("/prediction/time")
def get_predictions_by_time(
    start: str = Query(..., description="Start time in ISO format"),
    end: str = Query(..., description="End time in ISO format"),
    db: Session = Depends(get_db),
    user_id: int = Depends(get_current_user)
):
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    return queries.get_predictions_by_time(db, start_dt, end_dt)


@app.delete("/prediction/{uid}")
def delete_prediction(uid: str, db: Session = Depends(get_db), credentials: HTTPBasicCredentials = Depends(security)):
    user_id = verify_credentials(credentials, db)

    session = queries.get_prediction_by_uid(db, uid)
    if not session:
        raise HTTPException(status_code=404, detail="Prediction not found")
    if session.user_id != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized access to this prediction")
    # Delete image files if they exist
    for path in [session.original_image, session.predicted_image]:
        if os.path.exists(path):
            os.remove(path)

    # Delete from database
    queries.delete_detection_objects_by_uid(db, uid)
    queries.delete_prediction_session(db, uid)

    return "Successfully Deleted"

        
@app.get("/predictions/label/{label}")
def get_predictions_by_label(label: str, db: Session = Depends(get_db), user_id: int = Depends(get_current_user)):
    """
    Get prediction sessions containing objects with specified label
    """
    if label not in labels:
        raise HTTPException(status_code=400, detail="Invalid label")
    
    return queries.get_predictions_by_label(db, label)


@app.get("/predictions/score/{min_score}")
def get_predictions_by_score(
    min_score: float = Path(...),
    db: Session = Depends(get_db),
    user_id: int | None = Depends(get_current_user)
):
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    if min_score < 0 or min_score > 1:
        raise HTTPException(status_code=400, detail="Invalid score")
    results = queries.get_predictions_by_score(db, user_id, min_score)
    return [
        {
            "label": obj.label,
            "score": obj.score,
            "box": obj.box,
            "prediction_uid": obj.prediction_uid
        }
        for obj in results
    ]



@app.get("/image/{type}/{filename}")
def get_image(type: str, filename: str,user_id: int = Depends(get_current_user)):
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
def get_prediction_image(
    uid: str,
    request: Request,
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
    db: Session = Depends(get_db),
):
    user_id = verify_credentials(credentials, db)
    prediction = queries.get_prediction_by_uid(db, uid)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    if prediction.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    image_path = prediction.predicted_image
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")

    # Guess media type from file extension
    if image_path.endswith(".png"):
        media_type = "image/png"
    elif image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
        media_type = "image/jpeg"
    else:
        media_type = "application/octet-stream"  # safest fallback

    return FileResponse(image_path, media_type=media_type)


@app.get("/health")
def health():
    """
    Health check endpoint
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080,reload=True)
