from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import os
from models.schema import User, UserPreference, UserQuery, init_db

# Initialize database connection
engine = init_db()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# User operations
def create_or_get_user(db, user_id, email):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        user = User(id=user_id, email=email)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

def update_user_preferences(db, user_id, preferences):
    user_pref = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
    if not user_pref:
        user_pref = UserPreference(user_id=user_id, **preferences)
        db.add(user_pref)
    else:
        for key, value in preferences.items():
            if hasattr(user_pref, key):
                setattr(user_pref, key, value)
    db.commit()
    db.refresh(user_pref)
    return user_pref

def get_user_preferences(db, user_id):
    user_pref = db.query(UserPreference).filter(UserPreference.user_id == user_id).first()
    return user_pref

# Query history operations
def save_user_query(db, user_id, query_text, filters=None, results=None):
    query = UserQuery(
        user_id=user_id,
        query_text=query_text,
        filters=filters,
        results=results
    )
    db.add(query)
    db.commit()
    db.refresh(query)
    return query

def get_user_query_history(db, user_id, limit=10):
    queries = db.query(UserQuery).filter(
        UserQuery.user_id == user_id
    ).order_by(UserQuery.created_at.desc()).limit(limit).all()
    return queries