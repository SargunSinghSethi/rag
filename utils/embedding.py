import os
from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def create_embedding(text):
    """Create an embedding vector for a text using OpenAI API"""
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        return None

def format_gpu_for_embedding(gpu_data):
    """Format GPU data for embedding generation"""
    # Create a detailed description for better semantic search
    description = f"""
    GPU: {gpu_data.get('gpu_description', 'Unknown')} 
    Resource class: {gpu_data.get('resource_class', 'N/A')}
    vCPUs: {gpu_data.get('vcpus', 'N/A')}
    RAM: {gpu_data.get('ram', 'N/A')} GB
    Region: {gpu_data.get('region', 'N/A')}
    Country: {gpu_data.get('country', 'N/A')}
    Price per hour: ${gpu_data.get('price_per_hour', 'N/A')}
    Price per month: ${gpu_data.get('price_per_month', 'N/A')}
    Spot price: ${gpu_data.get('price_per_spot', 'N/A')}
    """
    return description.strip()

def embed_gpu_dataset(csv_path, save_path=None):
    """
    Process a CSV of GPU data and create embeddings
    
    Args:
        csv_path: Path to the CSV file with GPU data
        save_path: Path to save embeddings (optional)
        
    Returns:
        DataFrame with original data and embeddings
    """
    # Load GPU data
    df = pd.read_csv(csv_path)
    
    # Create descriptions for embedding
    df['embedding_text'] = df.apply(lambda row: format_gpu_for_embedding(row), axis=1)
    
    # Generate embeddings (warning: this makes API calls for each row)
    print(f"Generating embeddings for {len(df)} GPUs...")
    df['embedding'] = df['embedding_text'].apply(create_embedding)
    
    # Remove rows with failed embeddings
    df = df.dropna(subset=['embedding'])
    
    # Save embeddings if path provided
    if save_path:
        # Save embeddings separately as JSON since they're nested lists
        embeddings_dict = {
            row['resource_name'] if 'resource_name' in row else f"gpu_{i}": row['embedding'] 
            for i, row in df.iterrows()
        }
        with open(save_path, 'w') as f:
            json.dump(embeddings_dict, f)
        print(f"Saved {len(embeddings_dict)} embeddings to {save_path}")
    
    return df

def find_similar_gpus(query_embedding, gpu_embeddings, top_k=5):
    """
    Find similar GPUs by comparing embeddings using cosine similarity
    
    Args:
        query_embedding: Embedding vector for the query
        gpu_embeddings: Dict of GPU name -> embedding vector
        top_k: Number of results to return
        
    Returns:
        List of tuples (gpu_name, similarity_score)
    """
    
    
    # Convert query to numpy array
    query_embedding = np.array(query_embedding).reshape(1, -1)
    
    results = []
    for gpu_name, embedding in gpu_embeddings.items():
        # Convert to numpy array
        gpu_embedding = np.array(embedding).reshape(1, -1)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(query_embedding, gpu_embedding)[0][0]
        results.append((gpu_name, similarity))
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top K
    return results[:top_k]

if __name__ == "__main__":
    # Example usage
    CSV_PATH = "data/gpu_prices.csv"
    EMBEDDINGS_PATH = "data/gpu_embeddings.json"
    
    # Check if file exists
    if os.path.exists(CSV_PATH):
        # Generate and save embeddings
        df_with_embeddings = embed_gpu_dataset(CSV_PATH, EMBEDDINGS_PATH)
        print(f"Created embeddings for {len(df_with_embeddings)} GPUs")
    else:
        print(f"CSV file not found: {CSV_PATH}")