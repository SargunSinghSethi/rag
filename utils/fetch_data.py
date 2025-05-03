# utils/fetch_data.py

import requests
import csv
import os

REGIONS = [
    "us-east-at-1",
    "ap-south-mum-1",
    "ap-south-noi-1"
]

BASE_URL = "https://customer.acecloudhosting.com/api/v1/pricing?is_gpu=true&resource=instances&region="
CSV_PATH = "data/gpus.csv"

def fetch_and_store_gpu_data():
    os.makedirs("data", exist_ok=True)
    with open(CSV_PATH, mode='w', newline='') as file:
        writer = None
        for region in REGIONS:
            url = BASE_URL + region
            response = requests.get(url)
            if response.status_code == 200:
                json_data = response.json().get("data", [])
                if not writer and json_data:
                    headers = list(json_data[0].keys()) + (["region"] if "region" not in json_data[0] else [])
                    writer = csv.DictWriter(file, fieldnames=headers)
                    writer.writeheader()
                for item in json_data:
                    item["region"] = region
                    writer.writerow(item)
            else:
                print(f"Failed to fetch data from {region}: {response.status_code}")

if __name__ == "__main__":
    fetch_and_store_gpu_data()
