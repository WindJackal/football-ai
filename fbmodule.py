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
        Updates internal variable self.league for use in data-gathering functions.
        """
        print(f'Leagues: {self.league_names}\n')
        league = input('League: ')
        while league not in self.league_names:
            print('Invalid league given. Please try again')
            league = input('League: ')
        self.league = league
    
    def get_league_data(self):
        """
        Returns data for all teams in the league.
        """
        if len(self.league) == 0:
            print('You need to choose a league first.\n')
            return None
        shooting = pd.read_html(self.league_urls[self.league_names.index(self.league)], attrs={'id': 'stats_squads_shooting_for'}, flavor='lxml', header=1)[0]
        miscellaneous = pd.read_html(self.league_urls[self.league_names.index(self.league)], attrs={'id': 'stats_squads_misc_for'}, flavor='lxml', header=1)[0]
        shooting_relevant = shooting[['Squad', 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT']]
        misc_relevant = miscellaneous[['CrdY', 'CrdR', 'Fls']]
        df = shooting_relevant.join(misc_relevant)
        df['CrdY'] = round(df['CrdY'] / miscellaneous['90s'], 2)
        df['CrdR'] = round(df['CrdR'] / miscellaneous['90s'], 2)
        df['Fls'] = round(df['Fls'] / miscellaneous['90s'], 2)
        return df
    
    def get_match_data(self):
        """
        Returns match data for the current season of the chosen league.
        """
        if len(self.league) == 0:
            print('You need to choose a league first.\n')
            return None
        df = pd.read_csv(self.data_urls[self.league_names.index(self.league)])
        data = pd.DataFrame(df[['FTR', 'HS', 'AS', 'HST', 'AST', 'HY', 'AY', 'HR', 'AR', 'HF', 'AF']])
        data['HGS'] = round(df['FTHG'] / df['HS'], 2)
        data['AGS'] = round(df['FTAG'] / df['AS'], 2)
        data['HGST'] = round(df['FTHG'] / df['HST'], 2)
        data['AGST'] = round(df['FTAG'] / df['AST'], 2)

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data.dropna(inplace=True)
        return data
    
    def get_data(self):
        """
        Returns match and team data for use in Predictor class.
        """
        self.choose_league()
        league_data = self.get_league_data()
        match_data = self.get_match_data()
        return league_data, match_data
    
    def select_criterion(self):
        """
        Returns the optimal criterion for the chosen league, as defined by trial and error
        """
        if self.league in ['Championship', 'Portugal', 'Belgium', '2-Bundesliga', 'Ligue 2', 'Serie A', 'Segunda Division']:
            criterion = 'gini'
        else:
            criterion = 'entropy'
        return criterion

class Predictor():
    def __init__(self, team_data, match_data, criterion):
        self.team_data = team_data
        self.match_data = match_data
        self.criterion = criterion
        self.X_train = ''
        self.X_test = ''
        self.Y_train = ''
        self.Y_test = ''
        self.home_teams = []
        self.away_teams = []
    
    def split_match_data(self):
        """
        Splits match data into training and test sets
        """
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.match_data.drop('FTR', axis=1), self.match_data['FTR'], random_state=0)
    
    def hyperparameter_filtering(self):
        """
        Selects the 'best' hyperparameters for the Random Forest, based on a self implemented method
        """
        max_depths = list(range(1, 10))
        min_leafs = list(range(1, 10))
        min_splits = list(range(2, 10))
        best_acc = -np.inf
        best_depth = ''
        best_leaf = ''
        best_split = ''

        for depth in max_depths:
            clf = RandomForestClassifier(max_depth=depth, random_state=0, criterion=self.criterion)
            clf.fit(self.X_train, self.Y_train)
            score = clf.score(self.X_test, self.Y_test)
            if score > best_acc:
                best_acc = score
                best_depth = depth
        
        best_acc = -np.inf

        for leaf in min_leafs:
            clf = RandomForestClassifier(max_depth=best_depth, random_state=0, criterion=self.criterion, min_samples_leaf=leaf)
            clf.fit(self.X_train, self.Y_train)
            score = clf.score(self.X_test, self.Y_test)
            if score > best_acc:
                best_acc = score
                best_leaf = leaf
        
        best_acc = -np.inf

        for split in min_splits:
            clf = RandomForestClassifier(max_depth=best_depth, random_state=0, criterion=self.criterion, min_samples_leaf=best_leaf, min_samples_split=split)
            clf.fit(self.X_train, self.Y_train)
            score = clf.score(self.X_test, self.Y_test)
            if score > best_acc:
                best_acc = score
                best_split = split
        
        return best_depth, best_leaf, best_split

    def get_teams(self):
        """
        Loops endlessly to get teams for prediction. Enter 'end' when asked for the next home team to stop the loop.
        """
        x = 0
        while x < 1:
            home_team = input('Home Team: ')
            if home_team == 'end':
                x += 1
                break
            away_team = input('Away Team: ')
            self.home_teams.append(home_team)
            self.away_teams.append(away_team)
    
    def filter_data(self, home, away):
        """
        Returns a team's data formatted for prediction.
        """
        ht = self.team_data.query(f'Squad == "{home}"')
        at = self.team_data.query(f'Squad == "{away}"')
        HS = ht.get('Sh/90').values[0]
        AS = at.get('Sh/90').values[0]
        HST = ht.get('SoT/90').values[0]
        AST = at.get('SoT/90').values[0]
        HY = ht.get('CrdY').values[0]
        AY = at.get('CrdY').values[0]
        HR = ht.get('CrdR').values[0]
        AR = at.get('CrdR').values[0]
        HF = ht.get('Fls').values[0]
        AF = at.get('Fls').values[0]
        HGS = ht.get('G/Sh').values[0]
        AGS = at.get('G/Sh').values[0]
        HGST = ht.get('G/SoT').values[0]
        AGST = at.get('G/SoT').values[0]

        return [HS, AS, HST, AST, HY, AY, HR, AR, HF, AF, HGS, AGS, HGST, AGST]