import chromadb
from chromadb.utils import embedding_functions
import csv
import os
from dotenv import load_dotenv

load_dotenv()

# Path to your GPU CSV
CSV_PATH = "data/gpu_prices.csv"
CHROMA_COLLECTION_NAME = "gpu_prices"

# Initialize ChromaDB client
client = chromadb.Client()

# Use OpenAI embedding model
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),  # safer to use environment variable
    model_name="text-embedding-ada-002"
)

# Create or get the collection
collection = client.get_or_create_collection(
    name=CHROMA_COLLECTION_NAME,
    embedding_function=openai_ef
)

def index_gpu_data_from_csv(csv_path: str = CSV_PATH):
    if not os.path.exists(csv_path):
        return {"error": "CSV file not found."}

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        docs = []
        ids = []
        metadatas = []

        for i, row in enumerate(reader):
            doc = f"{row['gpu_description']} with {row['vcpus']} vCPUs and {row['ram']} GB RAM. Region: {row['region']}. Price: {row['price_per_hour']}."
            docs.append(doc)
            ids.append(str(i))  # Ensure string IDs
            metadatas.append(row)

        # Optional: clear existing data to avoid duplicates
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)

        collection.add(documents=docs, ids=ids, metadatas=metadatas)

    return {"message": "ChromaDB index refreshed successfully.", "entries": len(docs)}


def recommend_gpus_by_query(query: str, top_k: int = 5):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results
