import csv
import os
from utils.pricing import fetch_gpu_data_from_regions

CSV_PATH = "data/gpu_prices.csv"

def fetch_live_data():
    data = fetch_gpu_data_from_regions()

    if not data:
        return {"message": "No GPU data found to refresh.", "entries": 0}

    # Ensure new computed fields are present
    for entry in data:
        monthly_price = float(entry.get("price_per_month", 0) or 0)
        entry["price_per_half_year"] = round(monthly_price * 6, 2)
        entry["price_per_year"] = round(monthly_price * 12, 2)

    with open(CSV_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    return {"message": "Data refreshed successfully.", "entries": len(data)}


def get_all_gpu_data():
    if not os.path.exists(CSV_PATH):
        return []
    data = []
    with open(CSV_PATH, newline="") as file:
        reader = csv.DictReader(file)
        data = list(reader)
    return data
