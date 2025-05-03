import requests
import time
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "https://customer.acecloudhosting.com/api/v1/pricing?is_gpu=true&resource=instances&region="

def fetch_gpu_data_from_region(region):
    """Fetch GPU data from a specific region"""
    url = BASE_URL + region
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            json_data = response.json()
            if json_data and "data" in json_data and not json_data.get("error"):
                # Tag data with region if not already included
                data = json_data["data"]
                for item in data:
                    if "region" not in item:
                        item["region"] = region
                return data
        logger.warning(f"Failed to fetch data from {region}: {response.status_code}")
    except Exception as e:
        logger.error(f"Error fetching from {region}: {str(e)}")
    return []

def fetch_gpu_data_from_regions(regions=None):
    """Fetch GPU data from multiple regions with retries"""
    if regions is None:
        regions = ["us-east-at-1", "ap-south-mum-1", "ap-south-noi-1"]
    
    combined_data = []
    
    for region in regions:
        # Try up to 3 times with exponential backoff
        for attempt in range(3):
            data = fetch_gpu_data_from_region(region)
            if data:
                combined_data.extend(data)
                break
            if attempt < 2:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying {region} in {wait_time} seconds...")
                time.sleep(wait_time)
    
    return combined_data

def filter_available_gpus(gpu_data):
    """Filter GPUs to include only those with valid prices"""
    return [
        gpu for gpu in gpu_data 
        if gpu.get("price_per_hour") and float(gpu.get("price_per_hour", 0)) > 0
    ]

def calculate_cost(gpu, hours=1, spot=False):
    """Calculate the cost for a specified number of hours"""
    price_field = "price_per_spot" if spot else "price_per_hour"
    hourly_rate = float(gpu.get(price_field, 0))
    return round(hourly_rate * hours, 2)