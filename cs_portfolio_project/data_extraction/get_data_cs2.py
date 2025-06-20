import requests
from datetime import datetime
import csv
from datetime import datetime, timedelta
import pandas as pd
from cs_portfolio_project.data_extraction.skins_lists import *
from dotenv import load_dotenv
import os
import ast
import json

# to get cookies and id, go to https://steamcommunity.com/market/pricehistory/?appid=730&market_hash_name=Prisma%202%20Case, inspect and "network", refresh, go to the request and find cookies

load_dotenv()

cookies = {
    "sessionid": os.getenv("SESSIONID"),
    "steamLoginSecure": os.getenv("STEAMLOGINSECURE")
}

def get_price_history(appid, market_hash_name, cookies):
    """
    Fetch price history for a CS2/CSGO skin from Steam Market with authentication.
    
    Args:
        appid (int): Application ID (730 for CS2/CSGO).
        market_hash_name (str): Market name of the item (e.g., "Prisma 2 Case").
        cookies (dict): Steam session cookies (e.g., {'sessionid': '...', 'steamLoginSecure': '...'}).
    
    Returns:
        pd.DataFrame: DataFrame with date, price, and volume, or None if failed.
    """
    url = "https://steamcommunity.com/market/pricehistory/"
    params = {
        "appid": appid,
        "market_hash_name": market_hash_name
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": f"https://steamcommunity.com/market/listings/{appid}/{market_hash_name}"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, cookies=cookies)
        response.raise_for_status()
        data = response.json()
        
        if data.get("success"):
            price_history = []
            # Debug: Print the first few date strings
            print("Sample date strings from API:")
            for i, entry in enumerate(data["prices"][:5]):
                print(f"Entry {i}: {entry[0]}")
            
            # Parse the data
            for entry in data["prices"]:
                date_str, price_str, volume_str = entry
                # Fix malformed date string (e.g., "Mar 31 2020 01: +0" -> "Mar 31 2020 01:00:00 +0000")
                if " +0" in date_str:
                    date_str = date_str.replace(": +0", ":00:00 +0000")
                date_obj = datetime.strptime(date_str, "%b %d %Y %H:%M:%S %z")
                price = float(price_str)
                volume = int(volume_str)
                price_history.append({
                    "date": date_obj,
                    "price": price,
                    "volume": volume
                })
            df = pd.DataFrame(price_history)
            return df.sort_values("date")
        else:
            print(f"API returned 'success': False for {market_hash_name}")
            print(f"Response: {response.text}")
            return None
    
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error: {e}")
        print(f"Response: {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except ValueError as e:
        print(f"Error decoding JSON: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None


appid = 730 #csgo

market_hash_name_list = agent_skins


def save_csv_from_list(market_hash_name_list):
    failed_items = []
    for item in market_hash_name_list:

        df_item = get_price_history(appid, item, cookies)
        # print(df_item)
        # break
        if df_item is not None and not df_item.empty:

            df_item=df_item[df_item['date']>= min(df_item['date']) + timedelta(days=60)]
            filename =f"{item.replace(' | ','_').replace(' ', '_').replace(':', '').replace('/', '_').replace('(', '').replace(')', '')}_price.csv"
            print(filename)
            output_dir = os.path.join("data", "raw", "market_prices", "agents")
            os.makedirs(output_dir, exist_ok=True)
            df_item.to_csv(os.path.join(output_dir, filename.lower()))
        else:
            print(f"Skipping {item} due to no data.")
            failed_items.append(item)
    print(failed_items)

save_csv_from_list(market_hash_name_list)




