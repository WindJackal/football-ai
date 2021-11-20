import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from warnings import filterwarnings

filterwarnings('ignore', '.*')

class Data():
    def __init__(self):
        self.league_names = ['Premier League', 'Championship', 'League One', 'League Two', 'La Liga', 'Segunda Division', 'Bundesliga', '2-Bundesliga', 
        'Serie A', 'Serie B', 'Ligue 1', 'Ligue 2', 'Eredivisie', 'Portugal', 'Scottish Premiership', 'Belgium', 'Turkey']
        self.data_urls = ['http://www.football-data.co.uk/mmz4281/2122/E0.csv', 'http://www.football-data.co.uk/mmz4281/2122/E1.csv', 
        'http://www.football-data.co.uk/mmz4281/2122/E2.csv', 'http://www.football-data.co.uk/mmz4281/2122/E3.csv', 
        'http://www.football-data.co.uk/mmz4281/2122/SP1.csv', 'http://www.football-data.co.uk/mmz4281/2122/SP2.csv', 'http://www.football-data.co.uk/mmz4281/2122/D1.csv', 
        'http://www.football-data.co.uk/mmz4281/2122/D2.csv', 'http://www.football-data.co.uk/mmz4281/2122/I1.csv', 'http://www.football-data.co.uk/mmz4281/2122/I2.csv', 
        'http://www.football-data.co.uk/mmz4281/2122/F1.csv', 'http://www.football-data.co.uk/mmz4281/2122/F2.csv', 'http://www.football-data.co.uk/mmz4281/2122/N1.csv', 
        'http://www.football-data.co.uk/mmz4281/2122/P1.csv', 'http://www.football-data.co.uk/mmz4281/2122/SC0.csv', 'https://www.football-data.co.uk/mmz4281/2122/B1.csv',
        'https://www.football-data.co.uk/mmz4281/2122/T1.csv']
        self.league_urls = ['http://fbref.com/en/comps/9/Premier-League-Stats', 'http://fbref.com/en/comps/10/Championship-Stats', 
        'http://fbref.com/en/comps/15/League-One-Stats', 'http://fbref.com/en/comps/16/League-Two-Stats', 'http://fbref.com/en/comps/12/La-Liga-Stats', 
        'http://fbref.com/en/comps/17/Segunda-Division-Stats', 'http://fbref.com/en/comps/20/Bundesliga-Stats', 
        'http://fbref.com/en/comps/33/2-Bundesliga-Stats', 'http://fbref.com/en/comps/11/Serie-A-Stats', 'http://fbref.com/en/comps/18/Serie-B-Stats', 
        'http://fbref.com/en/comps/13/Ligue-1-Stats', 'http://fbref.com/en/comps/60/Ligue-2-Stats', 'http://fbref.com/en/comps/23/Eredivisie-Stats', 
        'http://fbref.com/en/comps/32/Primeira-Liga-Stats', 'http://fbref.com/en/comps/40/Scottish-Premiership-Stats', 'https://fbref.com/en/comps/37/Belgian-First-Division-A-Stats',
        'https://fbref.com/en/comps/26/Super-Lig-Stats']
        self.league = ''
    
    def choose_league(self):
        """
        Returns the individual data of each team in the league as a DataFrame.
        """
        print(f'Leagues: {self.league_names}\n')
        league = input('League: ')
        while league not in self.league_names:
            print('Invalid league given. Please try again')
            league = input('League: ')
        self.league = league
    
    def get_league_data(self):
        if len(self.league) == 0:
            print('You need to choose a league first.\n')
            return None
        shooting = pd.read_html(self.league_urls[self.league_names.index(self.league)], attrs={'id': 'stats_squads_shooting_for'}, flavor='lxml', header=1)[0]
        miscellaneous = pd.read_html(self.league_urls[self.league_names.index(self.league)], attrs={'id': 'stats_squads_misc_for'}, flavor='lxml', header=1)[0]
        shooting_relevant = shooting[['Sh/90', 'SoT/90', 'G/Sh', 'G/SoT']]
        misc_relevant = miscellaneous[['CrdY', 'CrdR', 'Fls']]
        for i in misc_relevant.columns:
            shooting_relevant.insert(len(shooting_relevant.columns), i, misc_relevant.get(i))
        return shooting_relevant
    