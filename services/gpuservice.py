import csv
import os
import pandas as pd
from utils.pricing import fetch_gpu_data_from_regions
from services.chromadb_service import index_gpu_data_from_csv

CSV_PATH = "data/gpu_prices.csv"
FETCH_REGIONS = ["us-east-at-1", "ap-south-mum-1", "ap-south-noi-1"]

def ensure_directory():
    """Ensure data directory exists"""
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

def fetch_live_data():
    """Fetch live GPU data from all regions and store in CSV"""
    ensure_directory()
    data = fetch_gpu_data_from_regions(FETCH_REGIONS)

    if not data:
        return {"message": "No GPU data found to refresh.", "entries": 0}

    # Process and clean data
    for entry in data:
        # Convert numeric fields to proper types
        for field in ['vcpus', 'ram']:
            if field in entry:
                try:
                    entry[field] = int(entry[field])
                except (ValueError, TypeError):
                    pass  # Keep as-is if conversion fails
                    
        for field in ['price_per_hour', 'price_per_month', 'price_per_spot']:
            if field in entry:
                try:
                    entry[field] = float(entry[field])
                except (ValueError, TypeError):
                    pass  # Keep as-is if conversion fails
        
        # Add computed fields
        monthly_price = float(entry.get("price_per_month", 0) or 0)
        entry["price_per_half_year"] = round(monthly_price * 6, 2)
        entry["price_per_year"] = round(monthly_price * 12, 2)
        
        # Create a resource_name field for easier identification
        gpu_desc = entry.get("gpu_description", "").strip()
        resource_class = entry.get("resource_class", "").strip()
        entry["resource_name"] = f"{resource_class}-{gpu_desc}" if resource_class else gpu_desc

    # Write to CSV
    if data:
        df = pd.DataFrame(data)
        df.to_csv(CSV_PATH, index=False)
        
        # Update the vector database
        index_gpu_data_from_csv(CSV_PATH)
        
        return {"message": "Data refreshed successfully and indexed.", "entries": len(data)}
    else:
        return {"message": "No data retrieved from API", "entries": 0}

def get_all_gpu_data():
    """Get all GPU data from CSV"""
    if not os.path.exists(CSV_PATH):
        # Try fetching if file doesn't exist
        fetch_result = fetch_live_data()
        if fetch_result["entries"] == 0:
            return []
    
    try:
        df = pd.read_csv(CSV_PATH)
        return df.to_dict('records')
    except Exception:
        return []

def filter_gpu_data(filters=None):
    """Filter GPU data based on criteria"""
    if not os.path.exists(CSV_PATH):
        return []
        
    try:
        df = pd.read_csv(CSV_PATH)
        
        if filters:
            if filters.get("region"):
                df = df[df["region"] == filters["region"]]
            if filters.get("max_price") is not None:
                df = df[df["price_per_hour"] <= float(filters["max_price"])]
            if filters.get("min_ram") is not None:
                df = df[df["ram"] >= float(filters["min_ram"])]
            if filters.get("min_vcpus") is not None:
                df = df[df["vcpus"] >= int(filters["min_vcpus"])]
                
        return df.to_dict('records')
    except Exception as e:
        print(f"Error filtering GPU data: {e}")
        return []

def get_unique_values(field):
    """Get unique values for a specific field"""
    if not os.path.exists(CSV_PATH):
        return []
        
    try:
        df = pd.read_csv(CSV_PATH)
        if field in df.columns:
            return df[field].dropna().unique().tolist()
        return []
    except Exception:
        return []