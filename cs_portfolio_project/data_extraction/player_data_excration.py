
from bs4 import BeautifulSoup
import requests
import csv

url = "https://steamcharts.com/app/730" # URL of the SteamCharts page for CS2

# Send a GET request to fetch the webpage content
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
response = requests.get(url, headers=headers)

if response.status_code != 200:
    print(f"Failed status code: {response.status_code}")
    exit()

soup = BeautifulSoup(response.content, "html.parser")

table = soup.find("table")  #Look for the first table on the page
months = []
avg_players = []

# Iterate over each row in the table
for row in table.find_all("tr")[1:]:  #skip header
    columns = row.find_all("td")
    if len(columns) >= 2: 
        month = columns[0].text.strip()
        avg_player = columns[1].text.strip()
        months.append(month)
        avg_players.append(avg_player)


for month, avg_player in zip(months, avg_players):
    print(f"{month}\t\t{avg_player}")

with open("cs2_players_monthly.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Month", "Avg. Players"])  
    writer.writerows(zip(months, avg_players))

print("\nData saved to 'cs2_players.csv'")