# routes/gpt_routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import csv
import openai

# Load your OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

router = APIRouter()

def load_gpu_data():
    data = []
    with open("data/gpu_prices.csv", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

class GPTQuery(BaseModel):
    question: str
    region: str = None
    max_price: float = None

@router.post("/query", summary="Ask GPT about GPU recommendations")
async def query_gpus_with_gpt(query: GPTQuery):
    # Load and filter data
    gpus = load_gpu_data()
    filtered = []
    for gpu in gpus:
        if query.region and gpu.get("region") != query.region:
            continue
        if query.max_price and float(gpu.get("price_per_hour", 0)) > query.max_price:
            continue
        filtered.append(gpu)

    if not filtered:
        raise HTTPException(status_code=404, detail="No GPUs matched your filters.")

    # Prepare context string
    context = "\n".join([
        f"{gpu['resource_name']}: {gpu['gpu_description']}, ${gpu['price_per_hour']}/hr, {gpu['region']}"
        for gpu in filtered
    ])

    prompt = (
        f"You are an expert cloud infrastructure consultant. "
        f"Given the following GPU options:\n{context}\n"
        f"Answer the question: {query.question}"
    )

    # Call OpenAI
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You provide GPU recommendations based on data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    answer = resp.choices[0].message.content
    return {"answer": answer}

# Then in main.py, include:
# from routes.gpt_routes import router as gpt_router
# app.include_router(gpt_router, prefix="/gpt", tags=["GPT Queries"])
