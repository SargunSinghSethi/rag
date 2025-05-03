from fastapi import FastAPI
from dotenv import load_dotenv
from routers import gpus
from routes.gpt_routes import router as gpt_router

load_dotenv()

app = FastAPI(title="GPU Cost Optimizer & Recommender")

app.include_router(gpus.router, prefix="/gpus", tags=["GPU Data"])
app.include_router(gpt_router, prefix="/gpt", tags=["GPT Queries"])