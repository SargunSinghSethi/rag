from fastapi import APIRouter
from services.gpuservice import get_all_gpu_data, fetch_live_data
from services.chromadb_service import index_gpu_data_from_csv, recommend_gpus_by_query

router = APIRouter()

@router.get("/all")
async def read_all_gpu_data():
    return get_all_gpu_data()

@router.get("/refresh")
async def refresh_data():
    return fetch_live_data()


@router.get("/gpus/index")
def index_chroma():
    return index_gpu_data_from_csv()

@router.get("/gpus/recommend")
def recommend_gpus(query: str, top_k: int = 5):
    return recommend_gpus_by_query(query, top_k)
