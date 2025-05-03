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
        
        # Create a resource_name field if it doesn't exist
        if "resource_name" not in entry or not entry["resource_name"]:
            resource_class = entry.get("resource_class", "").strip()
            entry["resource_name"] = f"{resource_class}" if resource_class else "unknown"
            
        # Add gpu_description field if using resource_name instead
        if "gpu_description" not in entry and "resource_name" in entry:
            entry["gpu_description"] = entry["resource_name"]

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
        # Ensure gpu_description exists
        if "gpu_description" not in df.columns and "resource_name" in df.columns:
            df["gpu_description"] = df["resource_name"]
        # Use country as region if region doesn't exist
        if "region" not in df.columns and "country" in df.columns:
            df["region"] = df["country"]
        return df.to_dict('records')
    except Exception as e:
        print(f"Error reading GPU data: {e}")
        return []

def filter_gpu_data(filters=None):
    """Filter GPU data based on criteria"""
    if not os.path.exists(CSV_PATH):
        return []
        
    try:
        df = pd.read_csv(CSV_PATH)
        
        # Ensure gpu_description exists
        if "gpu_description" not in df.columns and "resource_name" in df.columns:
            df["gpu_description"] = df["resource_name"]
            
        # Use country as region if region doesn't exist
        if "region" not in df.columns and "country" in df.columns:
            df["region"] = df["country"]
        
        if filters:
            # Handle region/country filtering
            if filters.get("region"):
                if "region" in df.columns:
                    df = df[df["region"] == filters["region"]]
                elif "country" in df.columns:
                    df = df[df["country"] == filters["region"]]
                    
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
        
        # Map field names if necessary
        field_mapping = {
            "region": ["region", "country"],  # Try region first, then country
            "gpu_description": ["gpu_description", "resource_name"]  # Try gpu_description first, then resource_name
        }
        
        fields_to_try = field_mapping.get(field, [field])
        
        for f in fields_to_try:
            if f in df.columns:
                return df[f].dropna().unique().tolist()
        
        return []
    except Exception as e:
        print(f"Error getting unique values: {e}")
        return []