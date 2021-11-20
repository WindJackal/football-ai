import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from warnings import filterwarnings

filterwarnings('ignore', '.*')

class League():
    def __init__(self):
        self.league_names = ['Premier League', 'Championship', 'League One', 'League Two', 'La Liga', 'Segunda Division', 'Bundesliga', '2-Bundesliga', 
        'Serie A', 'Serie B', 'Ligue 1', 'Ligue 2', 'Eredivisie', 'Portugal', 'Scottish Premiership', 'Belgium', 'Turkey']
    
    def get_league_data(self):
        """
        Returns the 
        """
