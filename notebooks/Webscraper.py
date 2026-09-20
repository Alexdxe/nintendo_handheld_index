#Packages:
import time
import requests
import pandas as pd
from io import StringIO




#Intailizing a Point of Reference for Webscraper: 
base_url = 'https://www.vgchartz.com/gamedb/'


#List of consoles to scrape & the number of games to scrape for each console: 
consoles = ['GB', 'GBA', 'DS', '3DS', 'NS']
games_per_console = 250

#Creating a function to scrape the games for each console:
def get_console_games(consoles, games_per_console):
    #create a list to store each console's data:
    all_dfs = []
    
    #Setting up an agent: 
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    #Iterating through each console to scrape all games: 
    for console in consoles: 
        #Sets up a url with the console name and number of games to scrape: 
        url = f'{base_url}/games.php?console={console}&results={games_per_console}&page=1'
        #Checking if the url is valid:
        response = requests.get(url, headers=headers)

        #IF our response is successful (status code 200):
        if response.status_code == 200:
            print(f'Scraping data for Nintendo {console}...')
            #turn the response into a open file object:
            html_file = StringIO(response.text)
            #Writing the responses as a Pandas Table: (adding lxml flavor to avoid warning message)
            table = pd.read_html(html_file, flavor="lxml")

            #Select the specific dataframe with the game data:
            game_df = table[6]

            #Ensuring only the last level of text is scraped for each column name: 
            if isinstance(game_df.columns, pd.MultiIndex):
                game_df.columns = [col[-1] for col in game_df.columns]
            #Cleaning wording of each game:
            if 'Game' in game_df.columns:
                game_df['Game'] = game_df['Game.1'].astype(str).str.replace('Read the review', '', case=False).str.strip()

            #Creating a new column to store the name of the scraped console: 
            game_df['Console_Scraped'] = console

            #Appending the dataframe to the list of all dataframes: 
            all_dfs.append(game_df)


        #If not a status code 200:
        else: 
            print(f'Failed to retrieve data for Nintendo {console}, status code: {response.status_code}')
    

    #Add a time buffer between when each console is scraped:
    time.sleep(5)


    #Combining all the dataframes into one: 
    combined_df = pd.concat(all_dfs, ignore_index=True)
    return combined_df 





#RUNNING THE SCRAPER:
if __name__ == "__main__":
    df = get_console_games(consoles, games_per_console)
    #Exporting the dataframe to a CSV file:
    df.to_csv('nintendo_handheld_games.csv', index=False)
    print("Data exported to nintendo_handheld_games.csv")
    