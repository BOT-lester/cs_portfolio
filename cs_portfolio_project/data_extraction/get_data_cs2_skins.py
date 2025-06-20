import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os
from cs_portfolio_project.data_extraction.skins_lists import *
from dotenv import load_dotenv
load_dotenv()

def get_price_history(appid, market_hash_name, cookies, retries=2):
    """Fetch price history from Steam Market API with retry for name variations."""
    url = "https://steamcommunity.com/market/pricehistory/"
    params = {"appid": appid, "market_hash_name": market_hash_name}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": f"https://steamcommunity.com/market/listings/{appid}/{market_hash_name}"
    }
    
    # Try original name and variations
    name_variations = [market_hash_name]
    if "(Holo/Foil)" in market_hash_name:
        name_variations.append(market_hash_name.replace("(Holo/Foil)", "(Holo-Foil)"))
    
    for attempt, name in enumerate(name_variations[:retries]):
        params["market_hash_name"] = name
        try:
            response = requests.get(url, params=params, headers=headers, cookies=cookies)
            response.raise_for_status()
            data = response.json()
            
            if data.get("success") and data.get("prices"):
                print(f"Sample date strings from API for {name}:")
                for i, entry in enumerate(data["prices"][:5]):
                    print(f"Entry {i}: {entry[0]}")
                
                price_history = []
                for entry in data["prices"]:
                    date_str, price_str, volume_str = entry
                    if " +0" in date_str:
                        date_str = date_str.replace(": +0", ":00:00 +0000")
                    date_obj = datetime.strptime(date_str, "%b %d %Y %H:%M:%S %z")
                    price = float(price_str)
                    volume = int(volume_str)
                    price_history.append({"date": date_obj, "price": price, "volume": volume})
                return pd.DataFrame(price_history), name  # Return DataFrame and successful name
            else:
                print(f"API failed for {name}: {response.text}")
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error for {name}: {e}")
            print(f"Response: {response.text}")
        except Exception as e:
            print(f"Error fetching {name}: {e}")
        
        if attempt < retries - 1:
            print(f"Retrying with variation... ({attempt + 1}/{retries})")
            time.sleep(1)
    
    print(f"All attempts failed for {market_hash_name}")
    return None, market_hash_name

def save_csv_from_skin_list(skin_list:list, cookies:dict, skin_list_name:str, appid=730):
    """Process a list of skins and save price history CSV for each condition."""
    conditions = ["Factory New", "Minimal Wear", "Field-Tested", "Well-Worn", "Battle-Scarred"]
    failed_items = []

    # Save in: data/raw/market_prices/<skin_list_name>/
    output_dir = os.path.join("data", "raw", "market_prices", skin_list_name)
    os.makedirs(output_dir, exist_ok=True)

    for skin in skin_list:
        for condition in conditions:
            market_hash_name = f"{skin} ({condition})"
            print(f"\nProcessing {market_hash_name}...")
            df_item, successful_name = get_price_history(appid, market_hash_name, cookies)

            if df_item is not None and not df_item.empty:
                # Filter to start 2 months after the earliest date
                df_item = df_item[df_item['date'] >= df_item['date'].min() + timedelta(days=60)]
                df_item['date'] = df_item['date'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S %z"))
                filename = f"{successful_name.replace(' | ', '_').replace(' ', '_')}_price_history.csv"
                # output_dir = os.path.join("data", "raw", "market_prices", "agents")
                # os.makedirs(output_dir, exist_ok=True)
                df_item.to_csv(os.path.join(output_dir, filename.lower()), index=False)
                print(f"Saved data to {os.path.join(output_dir, filename)}")
            else:
                print(f"Skipping {market_hash_name} due to no data.")
                failed_items.append(market_hash_name)

            time.sleep(1)  # Avoid rate limiting

    if failed_items:
        print("\nFailed items:")
        for item in failed_items:
            print(item)
        print("\nCheck names manually on Steam Market and update the list if needed.")

# Example list of skins 
skin_list = agent_skins

#cookies
cookies = {
    "sessionid": os.getenv("SESSIONID"),
    "steamLoginSecure": os.getenv("STEAMLOGINSECURE")
}

save_csv_from_skin_list(skin_list, cookies,"ancien_collection_list")