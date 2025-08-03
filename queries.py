# repository/queries.py

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session
from models import PredictionSession, DetectionObjects
from typing import List, Optional
from datetime import datetime, timedelta
from models import User


def save_prediction_session(db: Session, uid: str, original_image: str, predicted_image: str, user_id: Optional[int] = None):
    prediction = PredictionSession(
        uid=uid,
        original_image=original_image,
        predicted_image=predicted_image,
        user_id=user_id,
        timestamp=datetime.now()
    )
    db.add(prediction)
    db.commit()


def save_detection_object(db: Session, prediction_uid: str, label: str, score: float, box: str):
    obj = DetectionObjects(
        prediction_uid=prediction_uid,
        label=label,
        score=score,
        box=box
    )
    db.add(obj)
    db.commit()


def get_prediction_by_uid(db: Session, uid: str) -> Optional[PredictionSession]:
    return db.query(PredictionSession).filter_by(uid=uid).first()


def delete_prediction_session(db: Session, uid: str):
    db.query(PredictionSession).filter_by(uid=uid).delete()
    db.commit()


def delete_detection_objects_by_uid(db: Session, uid: str):
    db.query(DetectionObjects).filter_by(prediction_uid=uid).delete()
    db.commit()

def get_all_labels(db: Session, user_id: int) -> list[dict]:
    results = (
        db.query(DetectionObjects.label, func.count(DetectionObjects.label))
        .join(PredictionSession, DetectionObjects.prediction_uid == PredictionSession.uid)
        .filter(PredictionSession.user_id == user_id)
        .group_by(DetectionObjects.label)
        .all()
    )
    return [{"label": label, "count": count} for label, count in results]


def get_predictions_by_time(db: Session, start: datetime, end: datetime) -> list[dict]:
    rows = (
        db.query(PredictionSession.uid, PredictionSession.timestamp)
        .filter(PredictionSession.timestamp.between(start, end))
        .all()
    )
    return [{"uid": uid, "timestamp": timestamp} for uid, timestamp in rows]

def get_prediction_count_last_week(db: Session) -> int:
    one_week_ago = datetime.now() - timedelta(days=7)
    return db.query(func.count(PredictionSession.uid)).filter(PredictionSession.timestamp >= one_week_ago).scalar()


def count_recent_predictions(db: Session, days: int = 7) -> int:
    """
    Count how many predictions were made in the last X days (default: 7).
    """
    since = datetime.utcnow() - timedelta(days=days)
    return db.query(func.count(PredictionSession.uid)).filter(PredictionSession.timestamp >= since).scalar()

def get_predictions_by_label(db: Session, label: str) -> List[dict]:
    """
    Return list of predictions (uid, timestamp) that contain the given label
    """
    results = (
        db.query(PredictionSession.uid, PredictionSession.timestamp)
        .join(DetectionObjects, DetectionObjects.prediction_uid == PredictionSession.uid)
        .filter(DetectionObjects.label == label)
        .distinct()
        .all()
    )
    return [{"uid": uid, "timestamp": timestamp} for uid, timestamp in results]

def get_predictions_by_min_score(db: Session, user_id: int, min_score: float) -> list[dict]:
    results = (
        db.query(DetectionObjects.label, func.count(DetectionObjects.id))
        .join(PredictionSession, DetectionObjects.prediction_uid == PredictionSession.uid)
        .filter(
            and_(
                DetectionObjects.score >= min_score,
                PredictionSession.user_id == user_id
            )
        )
        .group_by(DetectionObjects.label)
        .all()
    )
    return [{"label": label, "count": count} for label, count in results]

def get_predictions_by_score(db: Session, user_id: int, min_score: float) -> list[DetectionObjects]:
    return (
        db.query(DetectionObjects)
        .join(PredictionSession, DetectionObjects.prediction_uid == PredictionSession.uid)
        .filter(PredictionSession.user_id == user_id, DetectionObjects.score >= min_score)
        .all()
    )


def get_user_by_credentials(db: Session, username: str, password: str) -> User | None:
    """
    Fetch a user by username and password.
    Returns None if no match found.
    """
    return db.query(User).filter(User.username == username, User.pass_field == password).first()


def add_test_user(db: Session, username: str = "user", password: str = "pass"):
    """
    Insert a demo user for testing if not exists.
    """
    user = db.query(User).filter(User.username == username).first()
    if not user:
        test_user = User(username=username, pass_field=password)
        db.add(test_user)
        db.commit()

