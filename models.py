from sqlalchemy import Column, String, DateTime, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    pass_field = Column("pass", String)

    predictions = relationship("PredictionSession", back_populates="user")


class PredictionSession(Base):
    __tablename__ = 'prediction_sessions'
    
    uid = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    original_image = Column(String)
    predicted_image = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))

    user = relationship("User", back_populates="predictions")


class DetectionObjects(Base):
    __tablename__ = 'detection_objects'

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_uid = Column(String, ForeignKey("prediction_sessions.uid"))
    label = Column(String)
    score = Column(Float)
    box = Column(String)
