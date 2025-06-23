# cs_portfolio

### data_extraction folder
get_data_cs2.py to scrap skins data. Names of skins for a given category should be a list in skins_list. 
Put personnal cookies in .env to acces the steam market website.
To process data after scraping it use get_prices_assetclass.py
example 
python ./cs_portfolio_project/data_extraction/get_prices_assetclass.py --skin_type cases --smooth false --remove_active_drop true

### optimisation folder
How to use the code examples are in code_demonstration notebook