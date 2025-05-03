from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from pydantic import BaseModel

from services.gpuservice import get_all_gpu_data, fetch_live_data, filter_gpu_data, get_unique_values
from services.chromadb_service import recommend_gpus_by_query, index_gpu_data_from_csv
from services.llm_service import recommend_gpu_with_llm, explain_recommendation
from services.db_service import get_db, create_or_get_user, get_user_preferences, save_user_query, update_user_preferences, get_user_query_history

router = APIRouter()

class GPUFilterParams(BaseModel):
    region: Optional[str] = None
    max_price: Optional[float] = None
    min_ram: Optional[int] = None
    min_vcpus: Optional[int] = None

class RecommendationRequest(BaseModel):
    query: str
    filters: Optional[GPUFilterParams] = None
    user_id: Optional[str] = None
    save_history: bool = True

# Basic GPU data endpoints
@router.get("/all")
async def read_all_gpu_data():
    """Get all available GPU data from the local database"""
    data = get_all_gpu_data()
    if not data:
        raise HTTPException(status_code=404, detail="No GPU data available. Try refreshing.")
    return {"data": data, "count": len(data)}

@router.get("/refresh", status_code=200)
async def refresh_gpu_data():
    """Refresh GPU data from the source API"""
    result = fetch_live_data()
    if result["entries"] == 0:
        raise HTTPException(status_code=500, detail="Failed to fetch GPU data from API")
    return result

@router.get("/filter")
async def filter_gpus(
    region: Optional[str] = None,
    max_price: Optional[float] = None,
    min_ram: Optional[int] = None,
    min_vcpus: Optional[int] = None
):
    """Filter GPUs based on criteria"""
    filters = {}
    if region:
        filters["region"] = region
    if max_price is not None:
        filters["max_price"] = max_price
    if min_ram is not None:
        filters["min_ram"] = min_ram
    if min_vcpus is not None:
        filters["min_vcpus"] = min_vcpus
        
    data = filter_gpu_data(filters)
    return {"data": data, "count": len(data)}

@router.get("/regions")
async def get_regions():
    """Get available regions from the dataset"""
    regions = get_unique_values("region")
    return {"regions": regions}

# Vector search endpoints
@router.get("/index")
async def index_chroma():
    """Index GPU data in ChromaDB for vector search"""
    return index_gpu_data_from_csv()

@router.get("/vector-search")
async def vector_search(query: str, top_k: int = 5):
    """Search for GPUs using vector similarity"""
    results = recommend_gpus_by_query(query, top_k=top_k)
    return {"results": results, "count": len(results)}

# LLM-powered recommendation endpoints
@router.post("/recommend")
async def recommend_gpus(
    request: RecommendationRequest,
    db: Session = Depends(get_db)
):
    """Get LLM-powered GPU recommendations based on natural language query"""
    # Filter GPU data if needed
    filters = {}
    if request.filters:
        if request.filters.region:
            filters["region"] = request.filters.region
        if request.filters.max_price is not None:
            filters["max_price"] = request.filters.max_price
        if request.filters.min_ram is not None:
            filters["min_ram"] = request.filters.min_ram
        if request.filters.min_vcpus is not None:
            filters["min_vcpus"] = request.filters.min_vcpus
    
    # Get filtered data
    gpu_data = filter_gpu_data(filters)
    
    if not gpu_data:
        raise HTTPException(status_code=404, detail="No GPUs available with these filters")
    
    # Get user preferences if user_id provided
    user_preferences = None
    if request.user_id:
        user = create_or_get_user(db, request.user_id, f"user_{request.user_id}@example.com")
        user_preferences = get_user_preferences(db, request.user_id)
        
        # Save query to history if requested
        if request.save_history:
            save_user_query(
                db=db,
                user_id=request.user_id,
                query_text=request.query,
                filters=filters if filters else None
            )
    
    # Generate recommendation using LLM
    recommendation = recommend_gpu_with_llm(
        query=request.query,
        gpu_data=gpu_data,
        user_preferences=user_preferences
    )
    
    if "error" in recommendation:
        raise HTTPException(status_code=500, detail=recommendation["error"])
    
    # Return the full recommendation
    return recommendation

@router.post("/explain")
async def get_explanation(recommendation: dict, details: bool = False):
    """Get a plain text explanation of a GPU recommendation"""
    explanation = explain_recommendation(recommendation, details)
    return {"explanation": explanation}

# User preference endpoints
@router.post("/preferences")
async def set_user_preferences(
    preferences: dict,
    user_id: str,
    db: Session = Depends(get_db)
):
    """Set or update user preferences for recommendations"""
    user = create_or_get_user(db, user_id, f"user_{user_id}@example.com")
    updated_preferences = update_user_preferences(db, user_id, preferences)
    return {"message": "Preferences updated", "preferences": updated_preferences}

@router.get("/preferences/{user_id}")
async def get_preferences(user_id: str, db: Session = Depends(get_db)):
    """Get user preferences"""
    preferences = get_user_preferences(db, user_id)
    if not preferences:
        raise HTTPException(status_code=404, detail="No preferences found for this user")
    return preferences

@router.get("/history/{user_id}")
async def get_query_history(user_id: str, limit: int = 10, db: Session = Depends(get_db)):
    """Get user query history"""
    history = get_user_query_history(db, user_id, limit)
    return {"history": history, "count": len(history)}