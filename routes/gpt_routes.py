# routes/gpt_routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import os
import csv
import openai
from openai import OpenAI

# Load your OpenAI API key from environment variable
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

router = APIRouter()

def load_gpu_data():
    data = []
    try:
        with open("data/gpu_prices.csv", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
        return data
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"Error loading GPU data: {e}")
        return []

class GPTQuery(BaseModel):
    question: str
    region: str = None
    max_price: float = None

@router.post("/query", summary="Ask GPT about GPU recommendations")
async def query_gpus_with_gpt(query: GPTQuery):
    # Load and filter data
    gpus = load_gpu_data()
    
    if not gpus:
        raise HTTPException(status_code=404, detail="No GPU data available. Please refresh the data first.")
    
    filtered = []
    for gpu in gpus:
        # Check region/country filter
        if query.region:
            region_matched = False
            if "region" in gpu and gpu["region"] == query.region:
                region_matched = True
            elif "country" in gpu and gpu["country"] == query.region:
                region_matched = True
                
            if not region_matched:
                continue
                
        # Check price filter
        if query.max_price and "price_per_hour" in gpu:
            try:
                if float(gpu["price_per_hour"]) > query.max_price:
                    continue
            except (ValueError, TypeError):
                continue
                
        filtered.append(gpu)

    if not filtered:
        raise HTTPException(status_code=404, detail="No GPUs matched your filters.")

    # Prepare context string
    context_items = []
    for gpu in filtered:
        # Use resource_name as primary identifier
        gpu_name = gpu.get('resource_name', 'Unknown GPU')
        region = gpu.get('region', gpu.get('country', 'Unknown region'))
        price = gpu.get('price_per_hour', 'N/A')
        
        context_items.append(
            f"{gpu_name}: {gpu.get('operating_system', 'N/A')}, ${price}/hr, {region}"
        )
    
    context = "\n".join(context_items)

    prompt = (
        f"You are an expert cloud infrastructure consultant. "
        f"Given the following GPU options:\n{context}\n"
        f"Answer the question: {query.question}"
    )

    # Call OpenAI
    try:
        resp = client.chat.completions.create(
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