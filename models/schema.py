from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import os

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)  # This will be the Clerk user ID
    email = Column(String, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    preferences = relationship("UserPreference", back_populates="user", cascade="all, delete-orphan")
    queries = relationship("UserQuery", back_populates="user", cascade="all, delete-orphan")

class UserPreference(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    preferred_region = Column(String, nullable=True)
    workload_type = Column(String, nullable=True)  # e.g., "training", "inference", "rendering"
    budget_max = Column(Float, nullable=True)
    resource_preference = Column(JSON, nullable=True)  # e.g., {"min_vram": 24, "prefer_tensor_cores": true}
    
    user = relationship("User", back_populates="preferences")

class UserQuery(Base):
    __tablename__ = "user_queries"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey("users.id"))
    query_text = Column(String, nullable=False)
    filters = Column(JSON, nullable=True)  # e.g., {"region": "us-east", "max_price": 3.0}
    results = Column(JSON, nullable=True)  # Store the recommendation results
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="queries")

# Initialize database
def init_db():
    db_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost/gpu_recommender")
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return engine