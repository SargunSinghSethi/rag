import chromadb
from chromadb.utils import embedding_functions
import csv
import os
import json
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# Path to your GPU CSV
CSV_PATH = "data/gpu_prices.csv"
CHROMA_COLLECTION_NAME = "gpu_prices"
CHROMA_PERSIST_DIR = "data/chromadb"

# Initialize ChromaDB client with persistence
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# Use OpenAI embedding model
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-ada-002"
)

def get_collection():
    """Get or create the ChromaDB collection"""
    try:
        return client.get_collection(name=CHROMA_COLLECTION_NAME)
    except ValueError:
        return client.create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=openai_ef
        )

def format_gpu_document(row):
    """Create a rich text description of the GPU for better semantic search"""
    # Adapt to actual column names in the CSV
    gpu_name = row.get('resource_name', 'Unknown')
    
    # Create a detailed description for embedding
    description = f"""
    GPU: {gpu_name} 
    Resource class: {row.get('resource_class', 'N/A')}
    vCPUs: {row.get('vcpus', 'N/A')}
    RAM: {row.get('ram', 'N/A')} GB
    Country/Region: {row.get('country', row.get('region', 'N/A'))}
    Operating System: {row.get('operating_system', 'N/A')}
    Price per hour: ${row.get('price_per_hour', 'N/A')}
    Price per month: ${row.get('price_per_month', 'N/A')}
    Spot price: ${row.get('price_per_spot', 'N/A')}
    """
    return description.strip()

def index_gpu_data_from_csv(csv_path: str = CSV_PATH):
    """Index GPU data from CSV into ChromaDB"""
    if not os.path.exists(csv_path):
        return {"error": "CSV file not found."}

    collection = get_collection()
    
    # Optional: clear existing data to avoid duplicates
    try:
        collection.delete(where={})
    except Exception:
        pass  # Collection might be empty
        
    # Read CSV with pandas for better handling
    df = pd.read_csv(csv_path)
    
    # Skip rows with missing critical data
    df = df.dropna(subset=['price_per_hour'])
    
    # Map gpu_description if needed
    if "gpu_description" not in df.columns and "resource_name" in df.columns:
        df["gpu_description"] = df["resource_name"]
        
    # Map region if needed
    if "region" not in df.columns and "country" in df.columns:
        df["region"] = df["country"]
    
    # Convert DataFrame to documents for ChromaDB
    docs = []
    ids = []
    metadatas = []
    
    for i, row in df.iterrows():
        row_dict = row.to_dict()
        
        # Create document for semantic search
        doc = format_gpu_document(row_dict)
        
        # Clean metadata (ensure all values are string, number, or bool for ChromaDB)
        metadata = {}
        for k, v in row_dict.items():
            if pd.isna(v):
                continue
            if isinstance(v, (int, float)):
                metadata[k] = v
            else:
                metadata[k] = str(v)
        
        docs.append(doc)
        ids.append(f"gpu_{i}")
        metadatas.append(metadata)

    # Add to collection in batches if large
    batch_size = 100
    for i in range(0, len(docs), batch_size):
        collection.add(
            documents=docs[i:i+batch_size],
            ids=ids[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )

    return {"message": "ChromaDB index refreshed successfully.", "entries": len(docs)}

def recommend_gpus_by_query(query: str, filters=None, top_k: int = 5):
    """Query ChromaDB for GPU recommendations with optional filters"""
    collection = get_collection()
    
    # Prepare where clause for filtering
    where_clause = {}
    if filters:
        if filters.get("region"):
            # Try both region and country depending on what's available
            field_name = "region" if "region" in collection.peek()["metadatas"][0] else "country"
            where_clause[field_name] = filters["region"]
            
        if filters.get("max_price") is not None:
            where_clause["price_per_hour"] = {"$lte": float(filters["max_price"])}
            
        if filters.get("min_ram") is not None:
            where_clause["ram"] = {"$gte": float(filters["min_ram"])}
            
        if filters.get("min_vcpus") is not None:
            where_clause["vcpus"] = {"$gte": int(filters["min_vcpus"])}
    
    # Execute query with filters if provided
    if where_clause:
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause
        )
    else:
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )
    
    # Format results
    formatted_results = []
    if results["metadatas"] and results["metadatas"][0]:
        for i, metadata in enumerate(results["metadatas"][0]):
            item = {
                "id": results["ids"][0][i],
                "metadata": metadata,
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None
            }
            formatted_results.append(item)
    
    return formatted_results

def get_unique_regions():
    """Get unique regions from the collection"""
    collection = get_collection()
    try:
        results = collection.get()
        if results and results["metadatas"]:
            regions = set()
            for metadata in results["metadatas"]:
                # Look for both region and country fields
                region_field = "region" if "region" in metadata else "country"
                if region_field in metadata:
                    regions.add(metadata[region_field])
            return list(regions)
        return []
    except Exception:
        return []